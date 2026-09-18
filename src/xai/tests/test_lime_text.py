"""
test_lime_text.py

Runs hierarchical text LIME (lime_text.py) against the live BBC News
serving API, across several hardcoded, realistic multi-sentence
examples. Each example is checked independently: the target class for
LIME is fixed to whatever the model's own argmax prediction is for
THAT example's original document - so different examples can (and
likely will) explain different target classes.

Output is the full document text with two-tier highlighting (sentence
shading + n-gram highlighting within the top sentence), saved as a
standalone HTML file per example - not a bar chart, since a bar chart
of n-gram names doesn't show WHERE in the text the signal actually is.

Usage:
    1. Start the server first:
        cd demo_serving && ./run_server.sh
    2. In another terminal:
        python test_lime_text.py
"""

import os
import json

import requests

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from xai import lime_text as lime_text_module

BASE_URL = "http://localhost:8000"
MODEL_KEY = "tfidf"  # switch to "encoder" once that checkpoint is trained

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Multiple realistic, multi-sentence examples - each single-topic
# (unlike a single mixed-topic document), so the tool has to find
# which sentences/phrases WITHIN one coherent article are most
# predictive, not just spot an obviously off-topic sentence.
EXAMPLES = {
    "sport_example": (
        "The championship match ended in a thrilling victory for the home team. "
        "Fans erupted in cheers as the star striker scored the winning goal in extra time. "
        "The coach praised the squad for their resilience throughout the tournament. "
        "Ticket sales for next season have already surged following the win."
    ),
    "business_example": (
        "Quarterly earnings for the banking sector exceeded analyst expectations this week. "
        "Shares in several major firms rallied following the announcement. "
        "Investors remain cautiously optimistic about growth in the coming months. "
        "Economists point to strong consumer spending as a key driver."
    ),
    "tech_example": (
        "The company unveiled its latest smartphone at a highly anticipated launch event. "
        "The new device features an upgraded camera system and longer battery life. "
        "Analysts say the release could strengthen the firm's position in a competitive market. "
        "Pre-orders opened shortly after the announcement and demand has been strong."
    ),
}


def check_server_running():
    try:
        requests.get(f"{BASE_URL}/bbc_news/{MODEL_KEY}/describe", timeout=3)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not connect to the server at {BASE_URL}. "
            f"Start it first with: cd demo_serving && ./run_server.sh"
        ) from e


def get_original_prediction(text: str) -> dict:
    r = requests.post(f"{BASE_URL}/bbc_news/{MODEL_KEY}/predict", json={"text": text})
    r.raise_for_status()
    return r.json()


def make_fixed_class_predict_fn(target_class: str):
    """Wraps the API so predict_fn always returns the probability of the
    SAME target_class, regardless of what the argmax becomes mid-occlusion."""
    def predict_fn(text: str) -> float:
        r = requests.post(f"{BASE_URL}/bbc_news/{MODEL_KEY}/predict", json={"text": text})
        r.raise_for_status()
        return float(r.json()["predicted_probabilities"][target_class])
    return predict_fn


def save_json(result, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved JSON -> {path}")


def save_html(html_content, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write(html_content)
    print(f"  Saved HTML -> {path}")


def run_one_example(name, text):
    print(f"\n{'=' * 70}")
    print(f"Example: {name}")
    print(f"{'=' * 70}")
    print(f"Text: {text}\n")

    original = get_original_prediction(text)
    target_class = original["predicted_class"]
    print(f"Model's predicted class (fixed as LIME target): {target_class}")
    print(f"Predicted probabilities: {original['predicted_probabilities']}")

    predict_fn = make_fixed_class_predict_fn(target_class)

    result = lime_text_module.explain_text(
        text=text,
        predict_fn=predict_fn,
        top_k_sentences=1,
        include_bigrams=True,
    )

    print(f"\nSentence-level importance:")
    for s in sorted(result["sentences"], key=lambda r: abs(r["importance"]), reverse=True):
        print(f"  [{s['importance']:+.4f}] (idx {s['sentence_index']}) {s['sentence_text']}")

    text_html = lime_text_module.render_document_html(result, top_k_ngrams_per_sentence=8)
    full_page = lime_text_module.render_full_html_page(text_html, target_class, result["original_probability"])

    result_with_meta = {
        "example_name": name,
        "document_text": text,
        "target_class": target_class,
        "model_predicted_probabilities": original["predicted_probabilities"],
        **result,
    }
    save_json(result_with_meta, f"bbc_lime_text_{name}.json")
    save_html(full_page, f"bbc_lime_text_{name}.html")


def main():
    check_server_running()

    for name, text in EXAMPLES.items():
        run_one_example(name, text)

    print("\nDone.")


if __name__ == "__main__":
    main()