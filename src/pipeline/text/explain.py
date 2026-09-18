"""
explain.py

Runs hierarchical text LIME (xai/lime_text.py) on every synthetic
document, one at a time, against THAT document's own argmax-predicted
class (decided once from the model's actual prediction, then held
fixed for the whole occlusion analysis of that document - see
model_client.make_fixed_class_predict_fn's docstring for why). Different
documents can - and, since they're generated stratified across
categories, likely will - explain different target classes. There is
no single global target the way tabular's fixed positive_class works
for a binary problem, and no aggregation across documents is needed
here: every document is shown on its own in the report (report.py),
not rolled up into one ranking.

No PDP-equivalent step: PDP varies one named feature across a grid of
values, which doesn't map onto raw text.
"""

from .model_client import get_prediction, make_fixed_class_predict_fn
from xai import lime_text as lime_text_module


def explain_documents(
    synthetic_docs: list,
    predict_url: str,
    top_k_sentences: int = 1,
    include_bigrams: bool = True,
) -> list:
    """
    synthetic_docs: [{"text": str, "intended_category": str}, ...]

    Returns a list of per-document result dicts (lime_text.explain_text's
    own shape), each additionally tagged with document_text,
    intended_category (the category it was generated for),
    predicted_class (the model's own argmax - the fixed LIME target for
    that document), and predicted_probabilities (the full distribution).
    """
    per_doc_results = []
    for doc in synthetic_docs:
        text = doc["text"]
        predicted_class, predicted_probabilities = get_prediction(predict_url, text)
        predict_fn = make_fixed_class_predict_fn(predict_url, predicted_class)

        result = lime_text_module.explain_text(
            text=text,
            predict_fn=predict_fn,
            top_k_sentences=top_k_sentences,
            include_bigrams=include_bigrams,
        )
        result["document_text"] = text
        result["intended_category"] = doc["intended_category"]
        result["predicted_class"] = predicted_class
        result["predicted_probabilities"] = predicted_probabilities
        per_doc_results.append(result)

    return per_doc_results
