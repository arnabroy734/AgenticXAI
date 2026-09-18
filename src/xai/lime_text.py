"""
lime_text.py

Hierarchical, from-scratch text explainability for a FIXED target class
(the class must be chosen before calling this - typically the model's
own argmax prediction for the document - and held fixed throughout;
see the multiclass LIME discussion this module follows from).

Two-tier design, motivated by a real limitation of plain single-word
occlusion: individual words rarely carry meaning on their own, but
exhaustively testing every word/n-gram combination across a long
document is far too expensive. So:

    Tier 1 - Sentence level: leave-one-sentence-out occlusion across
        the whole document, to find which sentence(s) most support the
        predicted class.
    Tier 2 - N-gram level: within only the top sentence(s) from Tier 1,
        leave-one-out occlusion at the unigram and bigram level
        (stopwords excluded), to find which specific words/phrases
        within that sentence matter.

This is a simplified, exhaustive occlusion approach - not the full
Sampling and Occlusion (SOC) algorithm (Jin et al., 2020), which
additionally marginalizes over sampled context replacements. This
implementation is systematic leave-one-out at both tiers: cheaper, and
a reasonable first cut, but a documented simplification.

predict_fn contract:
    Takes a raw document string (the full text, already reconstructed
    with whatever occlusion has been applied), returns a single float:
    the model's predicted probability for the ONE target class fixed
    for this call. Never the argmax - always the same class throughout.

Every returned span (sentence or n-gram) includes character offsets so
a report generator can highlight the exact substring - n-gram offsets
are relative to the sentence's own text, per the two-tier output shape.
"""

import re
from typing import Callable, Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_SENTENCE_SPLIT_RE = re.compile(r"[^.!?]*[.!?]+(?:\s+|$)|[^.!?]+$")
_WORD_RE = re.compile(r"[A-Za-z']+")


def _split_sentences(text: str):
    """Returns a list of (sentence_text, start_char, end_char) tuples,
    offsets relative to the full document. Regex-based, no external
    tokenizer/data download required."""
    sentences = []
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        raw = match.group()
        if not raw.strip():
            continue
        stripped = raw.strip()
        start = text.index(stripped, match.start(), match.end() + 1)
        end = start + len(stripped)
        sentences.append((stripped, start, end))
    return sentences


def _tokenize_words_with_offsets(sentence_text: str):
    """Returns a list of (word, start_char, end_char) tuples, offsets
    relative to the sentence's own text."""
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(sentence_text)]


def _generate_ngram_candidates(tokens, include_bigrams: bool):
    """
    Builds unigram + bigram candidates from tokenized words:
      - Unigrams: every non-stopword word.
      - Bigrams: every adjacent pair, excluded only if BOTH words are
        stopwords (a stopword + content word pair is kept - e.g. "not
        good" - since such combinations, especially negation, can
        matter even though one word alone would be filtered out).
    """
    candidates = []

    for word, start, end in tokens:
        if word.lower() not in ENGLISH_STOP_WORDS:
            candidates.append({"ngram_text": word, "n": 1, "start_char_in_sentence": start, "end_char_in_sentence": end})

    if include_bigrams:
        for i in range(len(tokens) - 1):
            w1, s1, _ = tokens[i]
            w2, _, e2 = tokens[i + 1]
            if w1.lower() in ENGLISH_STOP_WORDS and w2.lower() in ENGLISH_STOP_WORDS:
                continue
            candidates.append({
                "ngram_text": f"{w1} {w2}",
                "n": 2,
                "start_char_in_sentence": s1,
                "end_char_in_sentence": e2,
            })

    return candidates


def _mask_span(text: str, start: int, end: int) -> str:
    """Removes a character span and collapses the resulting double space."""
    masked = text[:start] + text[end:]
    return re.sub(r"\s{2,}", " ", masked).strip()


def _safe_predict(predict_fn, text: str) -> Optional[float]:
    """Wraps predict_fn so a single failed call (e.g. an empty or
    degenerate document after occlusion) doesn't crash the whole
    explanation - returns None on failure, handled by the caller."""
    try:
        return float(predict_fn(text))
    except Exception:
        return None


def _reconstruct_without_sentence(sentences, exclude_index: int) -> str:
    remaining = [s[0] for i, s in enumerate(sentences) if i != exclude_index]
    return " ".join(remaining)


def _reconstruct_with_modified_sentence(sentences, sentence_index: int, modified_sentence_text: str) -> str:
    parts = [modified_sentence_text if i == sentence_index else s[0] for i, s in enumerate(sentences)]
    return " ".join(parts)


def explain_text(
    text: str,
    predict_fn: Callable[[str], float],
    top_k_sentences: int = 10,
    include_bigrams: bool = True,
) -> dict:
    """
    Explains predict_fn's output (a fixed target class's probability)
    for one document, via sentence-level then n-gram-level leave-one-
    out occlusion.

    Returns:
        {
            "original_probability": float,
            "sentences": [
                {"sentence_index": int, "sentence_text": str,
                 "start_char_in_doc": int, "end_char_in_doc": int,
                 "importance": float},
                ...
            ],
            "top_sentence_indices": [int, ...],
            "ngrams_by_sentence": {
                <sentence_index>: [
                    {"ngram_text": str, "n": 1 or 2,
                     "start_char_in_sentence": int, "end_char_in_sentence": int,
                     "importance": float},
                    ...
                ],
                ...
            },
        }

    importance for both tiers = original_probability - occluded_probability.
    Positive means removing that span DECREASED the target class's
    probability (i.e. that span supports the prediction).
    """
    original_prob = _safe_predict(predict_fn, text)
    if original_prob is None:
        raise ValueError("predict_fn failed on the original, unmodified text - cannot proceed.")

    sentences = _split_sentences(text)

    # --- Tier 1: sentence-level occlusion ---
    sentence_results = []
    for i, (sentence_text, start_char, end_char) in enumerate(sentences):
        if len(sentences) == 1:
            # Removing the only sentence leaves nothing meaningful to
            # score against - skip occlusion, importance undefined (0).
            importance = 0.0
        else:
            occluded_text = _reconstruct_without_sentence(sentences, i)
            occluded_prob = _safe_predict(predict_fn, occluded_text)
            importance = (original_prob - occluded_prob) if occluded_prob is not None else 0.0

        sentence_results.append({
            "sentence_index": i,
            "sentence_text": sentence_text,
            "start_char_in_doc": start_char,
            "end_char_in_doc": end_char,
            "importance": importance,
        })

    ranked = sorted(sentence_results, key=lambda r: abs(r["importance"]), reverse=True)
    top_sentence_indices = [r["sentence_index"] for r in ranked[:top_k_sentences]]

    # --- Tier 2: n-gram level occlusion, within top sentence(s) only ---
    ngrams_by_sentence = {}
    for sentence_index in top_sentence_indices:
        sentence_text = sentences[sentence_index][0]
        tokens = _tokenize_words_with_offsets(sentence_text)
        candidates = _generate_ngram_candidates(tokens, include_bigrams)

        ngram_results = []
        for candidate in candidates:
            masked_sentence = _mask_span(
                sentence_text, candidate["start_char_in_sentence"], candidate["end_char_in_sentence"]
            )
            occluded_doc = _reconstruct_with_modified_sentence(sentences, sentence_index, masked_sentence)
            occluded_prob = _safe_predict(predict_fn, occluded_doc)
            importance = (original_prob - occluded_prob) if occluded_prob is not None else 0.0

            ngram_results.append({
                "ngram_text": candidate["ngram_text"],
                "n": candidate["n"],
                "start_char_in_sentence": candidate["start_char_in_sentence"],
                "end_char_in_sentence": candidate["end_char_in_sentence"],
                "importance": importance,
            })

        ngrams_by_sentence[sentence_index] = sorted(ngram_results, key=lambda r: abs(r["importance"]), reverse=True)

    return {
        "original_probability": original_prob,
        "sentences": sentence_results,
        "top_sentence_indices": top_sentence_indices,
        "ngrams_by_sentence": ngrams_by_sentence,
    }


# ---------------------------------------------------------------------
# Rendering: two-tier highlighted text (sentence shading + n-gram
# highlighting within the top sentence(s)), as an HTML string. Lives
# here (not in a test script) since the real reporting pipeline will
# need this exact rendering later, not just this module's own tests.
# ---------------------------------------------------------------------

import html as _html


def _color_for_importance(importance: float, max_abs: float, alpha_min: float, alpha_max: float,
                           positive_rgb=(29, 158, 117), negative_rgb=(226, 75, 74)) -> str:
    if max_abs <= 0:
        alpha = 0.0
    else:
        alpha = alpha_min + (alpha_max - alpha_min) * min(abs(importance) / max_abs, 1.0)
    rgb = positive_rgb if importance >= 0 else negative_rgb
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.2f})"


def _select_non_overlapping_ngrams(ngrams: list, top_k: int) -> list:
    """
    Greedily selects the top_k highest-|importance| n-grams that don't
    character-overlap each other. Needed because unigrams and bigrams
    naturally overlap (e.g. "team" and "team won") - highlighting both
    would produce broken, conflicting spans in the rendered text.
    """
    ranked = sorted(ngrams, key=lambda r: abs(r["importance"]), reverse=True)
    selected = []
    claimed = []

    for ng in ranked:
        start, end = ng["start_char_in_sentence"], ng["end_char_in_sentence"]
        overlaps = any(not (end <= cs or start >= ce) for cs, ce in claimed)
        if not overlaps:
            selected.append(ng)
            claimed.append((start, end))
        if len(selected) >= top_k:
            break

    return selected


def _render_sentence_with_ngram_highlights(sentence_text: str, selected_ngrams: list, max_abs_ngram_importance: float) -> str:
    spans = sorted(selected_ngrams, key=lambda s: s["start_char_in_sentence"])
    pieces = []
    cursor = 0

    for s in spans:
        start, end = s["start_char_in_sentence"], s["end_char_in_sentence"]
        if start < cursor:
            continue  # safety net; _select_non_overlapping_ngrams already prevents this
        pieces.append(_html.escape(sentence_text[cursor:start]))
        color = _color_for_importance(s["importance"], max_abs_ngram_importance, alpha_min=0.25, alpha_max=0.65)
        piece_text = _html.escape(sentence_text[start:end])
        pieces.append(
            f'<span style="background-color:{color}; border-radius:3px; padding:1px 2px;" '
            f'title="{_html.escape(s["ngram_text"])}: {s["importance"]:+.4f}">{piece_text}</span>'
        )
        cursor = end

    pieces.append(_html.escape(sentence_text[cursor:]))
    return "".join(pieces)


def render_document_html(result: dict, top_k_ngrams_per_sentence: int = 8) -> str:
    """
    Renders the full document as an HTML string with two-tier
    highlighting:
        - every sentence gets a light background shade by its own
          sentence-level importance (green = supports the target
          class, red = detracts from it);
        - within the top sentence(s) only, the top non-overlapping
          n-grams get a stronger highlight on top of the sentence shade.
    """
    sentences = result["sentences"]
    top_indices = set(result["top_sentence_indices"])
    max_abs_sentence = max((abs(s["importance"]) for s in sentences), default=0.0) or 1.0

    rendered_sentences = []
    for s in sentences:
        idx = s["sentence_index"]
        sentence_text = s["sentence_text"]

        ngrams = result["ngrams_by_sentence"].get(idx, result["ngrams_by_sentence"].get(str(idx), []))
        if idx in top_indices and ngrams:
            max_abs_ngram = max((abs(ng["importance"]) for ng in ngrams), default=0.0) or 1.0
            selected = _select_non_overlapping_ngrams(ngrams, top_k=top_k_ngrams_per_sentence)
            inner_html = _render_sentence_with_ngram_highlights(sentence_text, selected, max_abs_ngram)
        else:
            inner_html = _html.escape(sentence_text)

        sentence_color = _color_for_importance(s["importance"], max_abs_sentence, alpha_min=0.05, alpha_max=0.18)
        rendered_sentences.append(
            f'<span style="background-color:{sentence_color}; padding:2px;" '
            f'title="sentence importance: {s["importance"]:+.4f}">{inner_html}</span>'
        )

    return " ".join(rendered_sentences)


def render_full_html_page(text_html: str, target_class: str, original_probability: float) -> str:
    """Wraps render_document_html's output in a standalone, styled HTML page with a legend."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Helvetica, Arial, sans-serif; font-size: 15px; line-height: 1.9;
          color: #1f2937; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
  h2 {{ color: #1E3A5F; }}
  .legend {{ font-size: 13px; color: #64748b; margin-bottom: 20px; }}
  .legend span {{ padding: 2px 8px; border-radius: 3px; margin-right: 6px; }}
</style>
</head>
<body>
  <h2>Text Explanation</h2>
  <p><strong>Target class:</strong> {_html.escape(target_class)} &nbsp; | &nbsp;
     <strong>Predicted probability:</strong> {original_probability:.4f}</p>
  <div class="legend">
    <span style="background-color:rgba(29,158,117,0.5);">supports prediction</span>
    <span style="background-color:rgba(226,75,74,0.5);">detracts from prediction</span>
    &nbsp; (darker = stronger effect; sentence shading is lighter, n-gram highlighting within the
    most important sentence is stronger)
  </div>
  <p>{text_html}</p>
</body>
</html>"""