"""
model_client.py

Everything about talking to a served text-classification model:
  - Step 1: parse an already-fetched /describe response - modality
    "text" and the category list (job_type must be "classification" -
    there's no regression equivalent for raw text).
  - Step 3-4: two predict helpers - get_prediction (the model's own
    argmax + full probability distribution for one document) and
    make_fixed_class_predict_fn (wraps that into a predict_fn(text)
    -> float fixed to ONE target class, for use during occlusion).
  - Step 2: generate + validate synthetic example documents via LLM,
    stratified evenly across every category so each class gets a
    comparable, substantial batch rather than an incidental mix.
"""

import logging
import re

import requests

from ..llm_client import call_llm_json

logger = logging.getLogger(__name__)

# Shared across every get_prediction call in this process, for the same
# reason as tabular/model_client.py's make_predict_fn: without HTTP
# keep-alive/connection reuse, occlusion (one call per sentence, per
# n-gram, per document) opens a fresh TCP connection every time, which
# can exhaust the container's ephemeral port range under load.
_session = requests.Session()

MIN_SENTENCES_PER_DOCUMENT = 2
MAX_TOKENS_PER_DOCUMENT = 512
# ~0.75 words per token is the standard rule of thumb for English text
# (e.g. OpenAI's own tokenizer guidance: "100 tokens ~= 75 words") -
# used here to turn a token budget into a word count an LLM prompt can
# actually be instructed with.
WORDS_PER_TOKEN = 0.75
MAX_WORDS_PER_DOCUMENT = round(MAX_TOKENS_PER_DOCUMENT * WORDS_PER_TOKEN)  # 384
# Sanity floor, not from any explicit spec - guarantees "at least 2
# sentences" can't be satisfied by two near-empty fragments.
MIN_WORDS_PER_DOCUMENT = 20

_SENTENCE_COUNT_RE = re.compile(r"[^.!?]*[.!?]+")


# ---------------------------------------------------------------------
# Step 1: parse /describe
# ---------------------------------------------------------------------

def parse_describe_body(describe_body: dict, base_url: str, model_key: str) -> dict:
    """
    Returns:
        {
            "job_type": "classification",
            "model_name": str,
            "categories": [...],
            "predict_url": "<base_url>/<model_key>/predict",
        }
    """
    job_type = describe_body["job_type"]
    if job_type != "classification":
        raise ValueError(
            f"Text modality backend only supports classification, got job_type='{job_type}' "
            f"for model '{model_key}'."
        )

    categories = describe_body.get("categories")
    if not categories:
        raise ValueError(f"Text classification model '{model_key}' did not declare 'categories' in /describe.")

    return {
        "job_type": job_type,
        "model_name": describe_body["model_name"],
        "categories": categories,
        "predict_url": f"{base_url}/{model_key}/predict",
    }


# ---------------------------------------------------------------------
# Step 3-4: prediction helpers
# ---------------------------------------------------------------------

def get_prediction(predict_url: str, text: str) -> tuple:
    """Returns (predicted_class, predicted_probabilities) for one document -
    predicted_class is the model's own argmax over predicted_probabilities."""
    r = _session.post(predict_url, json={"text": text})
    r.raise_for_status()
    body = r.json()
    return body["predicted_class"], body["predicted_probabilities"]


def make_fixed_class_predict_fn(predict_url: str, target_class: str):
    """
    predict_fn(text) -> float: probability of target_class, fixed for
    the whole occlusion analysis of ONE document. target_class is the
    model's own argmax prediction for that document, decided once up
    front (see explain.py) and never re-computed mid-occlusion - a
    perturbation that would flip the argmax must not make the
    explanation track a moving target.
    """
    def predict_fn(text: str) -> float:
        _, prob_dict = get_prediction(predict_url, text)
        return float(prob_dict[target_class])

    return predict_fn


# ---------------------------------------------------------------------
# Step 2: synthetic document generation, stratified by category
# ---------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_count(text: str) -> int:
    return len([m for m in _SENTENCE_COUNT_RE.findall(text) if m.strip()])


def _is_valid_document(text: str) -> bool:
    word_count = _word_count(text)
    return (
        MIN_WORDS_PER_DOCUMENT <= word_count <= MAX_WORDS_PER_DOCUMENT
        and _sentence_count(text) >= MIN_SENTENCES_PER_DOCUMENT
    )


def _build_prompt(category: str, num_docs: int) -> tuple:
    system_prompt = (
        "You are a synthetic news-article generator. You produce realistic, self-contained "
        "news article excerpts, each clearly belonging to ONE specific category. Respond ONLY "
        'with a JSON object of the form {"documents": ["...", "...", ...]} with no extra '
        "commentary, no markdown fences.\n\n"
        "Every example must satisfy ALL of the following:\n"
        f"- At least {MIN_SENTENCES_PER_DOCUMENT} complete sentences.\n"
        f"- No more than {MAX_WORDS_PER_DOCUMENT} words long (this keeps each example under "
        f"roughly {MAX_TOKENS_PER_DOCUMENT} tokens).\n"
        f"- Written entirely about the category \"{category}\" - not a mix of topics."
    )
    user_prompt = (
        f'Generate exactly {num_docs} distinct news article excerpts, ALL about the category '
        f'"{category}", as a JSON object with a single key "documents" containing a list of '
        f"{num_docs} strings. Each excerpt must be self-contained, realistic, and clearly about "
        f'"{category}" specifically - vary the angle/subtopic across examples so they are not '
        f"repetitive."
    )
    return system_prompt, user_prompt


def _call_llm_for_documents(category: str, num_docs: int) -> list:
    system_prompt, user_prompt = _build_prompt(category, num_docs)
    result = call_llm_json(system_prompt, user_prompt)
    return result.get("documents", [])


def _generate_for_category(category: str, num_docs: int, max_retries: int) -> list:
    valid_docs = []

    for attempt in range(1, max_retries + 1):
        remaining = num_docs - len(valid_docs)
        if remaining <= 0:
            break

        logger.info(
            "Category '%s' attempt %d/%d: requesting %d more document(s)",
            category, attempt, max_retries, remaining,
        )
        candidates = _call_llm_for_documents(category, remaining)

        attempt_valid = [d for d in candidates if isinstance(d, str) and _is_valid_document(d)]
        valid_docs.extend(attempt_valid)
        logger.info(
            "Category '%s': %d/%d candidate(s) passed validation. Total valid so far: %d/%d",
            category, len(attempt_valid), len(candidates), len(valid_docs), num_docs,
        )

    if len(valid_docs) < num_docs:
        logger.warning(
            "Category '%s': only %d/%d synthetic documents passed validation after %d attempts. "
            "Proceeding with the %d collected instead of discarding them.",
            category, len(valid_docs), num_docs, max_retries, len(valid_docs),
        )

    return valid_docs


def _stratified_counts(num_docs: int, num_categories: int) -> list:
    """Splits num_docs as evenly as possible across num_categories - e.g.
    100 docs / 4 categories -> [25, 25, 25, 25]. A remainder (num_docs
    not evenly divisible) is spread one-per-category starting from the
    first, so the total always adds up to exactly num_docs."""
    base = num_docs // num_categories
    remainder = num_docs % num_categories
    return [base + 1 if i < remainder else base for i in range(num_categories)]


def generate_synthetic_documents(categories: list, num_docs: int = 100, max_retries: int = 3) -> list:
    """
    Generates num_docs synthetic news-article excerpts, stratified as
    evenly as possible across every category (e.g. 100 documents over
    4 categories -> 25 per category), each validated against the
    word/sentence-count constraints above - an invalid document is
    dropped, not the whole batch, mirroring
    tabular/model_client.py's generate_synthetic_data.

    Returns [{"text": str, "intended_category": str}, ...]. Raises only
    if zero valid documents resulted overall.
    """
    counts = _stratified_counts(num_docs, len(categories))

    all_docs = []
    for category, count in zip(categories, counts):
        if count <= 0:
            continue
        docs = _generate_for_category(category, count, max_retries)
        all_docs.extend({"text": d, "intended_category": category} for d in docs)

    if not all_docs:
        raise RuntimeError(
            f"No valid synthetic documents could be generated for any category after "
            f"{max_retries} attempt(s) each. Cannot proceed with zero documents."
        )

    return all_docs
