"""
explain.py — LIME-based explainability for the Fake News Detection model.

LIME (Local Interpretable Model-agnostic Explanations) perturbs the input
text, observes prediction changes, and fits a local linear model to identify
which words most strongly influence the prediction.

Reference: https://github.com/marcotcr/lime
"""

import numpy as np
from sklearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer
from data_loader import clean_text

# ---------------------------------------------------------------------------
# Class labels (must align with model.py)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["REAL", "FAKE"]

# ---------------------------------------------------------------------------
# LIME predict wrapper
# ---------------------------------------------------------------------------

def _make_predict_proba(pipeline: Pipeline):
    """
    Return a predict_proba-compatible function for LIME.

    PassiveAggressiveClassifier has no predict_proba, so we convert
    the signed decision function values through a sigmoid to get
    class probability estimates.
    """
    def predict_proba(texts):
        cleaned = [clean_text(t) for t in texts]
        # decision_function returns a score per sample
        scores = pipeline.decision_function(cleaned)  # shape: (n_samples,)
        # Sigmoid converts score → P(FAKE)
        prob_fake = 1 / (1 + np.exp(-scores))
        prob_real = 1 - prob_fake
        # Stack into (n_samples, 2): col0=REAL, col1=FAKE
        return np.column_stack([prob_real, prob_fake])

    return predict_proba


# ---------------------------------------------------------------------------
# Core explanation function
# ---------------------------------------------------------------------------

def explain_prediction(
    pipeline: Pipeline,
    text: str,
    num_features: int = 12,
    num_samples: int = 500,
) -> dict:
    """
    Generate a LIME explanation for a single news article prediction.

    Args:
        pipeline:     trained sklearn Pipeline
        text:         raw input news text
        num_features: number of top words to highlight
        num_samples:  LIME perturbation samples (higher = more accurate, slower)

    Returns:
        dict:
          words_positive  — list of (word, weight) tuples supporting FAKE
          words_negative  — list of (word, weight) tuples supporting REAL
          all_features    — full sorted list of (word, weight)
          html            — raw LIME HTML visualization (optional use)
          lime_exp        — the raw LimeExplanation object
    """
    cleaned = clean_text(text)
    if not cleaned.strip():
        return {
            "words_positive": [],
            "words_negative": [],
            "all_features": [],
            "html": "",
            "lime_exp": None,
        }

    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES,
        split_expression=r"\b\w+\b",    # word tokenization
        bow=True,                        # bag-of-words mode
        random_state=42,
    )

    predict_proba_fn = _make_predict_proba(pipeline)

    exp = explainer.explain_instance(
        text_instance=cleaned,
        classifier_fn=predict_proba_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=(0, 1),                   # explain both classes
    )

    # Extract feature weights for FAKE class (index 1)
    features = exp.as_list(label=1)  # [(word, weight), ...]

    # Split into positive (pro-FAKE) and negative (pro-REAL) influences
    words_positive = [(w, round(v, 4)) for w, v in features if v > 0]
    words_negative = [(w, round(abs(v), 4)) for w, v in features if v < 0]

    # Sort by absolute impact
    words_positive.sort(key=lambda x: x[1], reverse=True)
    words_negative.sort(key=lambda x: x[1], reverse=True)

    # Generate LIME HTML (can be embedded in an iframe or saved)
    try:
        html = exp.as_html()
    except Exception:
        html = ""

    return {
        "words_positive": words_positive,   # push toward FAKE
        "words_negative": words_negative,   # push toward REAL
        "all_features": features,
        "html": html,
        "lime_exp": exp,
    }


# ---------------------------------------------------------------------------
# Highlight words in original text
# ---------------------------------------------------------------------------

def highlight_text(text: str, explanation: dict) -> list[dict]:
    """
    Annotate each token in the text with its LIME influence score.

    Returns a list of token dicts:
      { 'word': str, 'score': float, 'direction': 'fake'|'real'|'neutral' }

    Designed for rendering word-level highlighting in Streamlit.
    """
    pos_words = {w.lower(): v for w, v in explanation["words_positive"]}
    neg_words = {w.lower(): v for w, v in explanation["words_negative"]}

    # Split text preserving spaces for faithful reconstruction
    import re
    tokens = re.findall(r"\b\w+\b|\W+", text)

    annotated = []
    for token in tokens:
        key = token.lower().strip()
        if key in pos_words:
            annotated.append({
                "word": token,
                "score": pos_words[key],
                "direction": "fake",
            })
        elif key in neg_words:
            annotated.append({
                "word": token,
                "score": neg_words[key],
                "direction": "real",
            })
        else:
            annotated.append({
                "word": token,
                "score": 0.0,
                "direction": "neutral",
            })

    return annotated


def build_highlighted_html(annotated_tokens: list[dict]) -> str:
    """
    Render annotated tokens as an HTML string with inline highlighted spans.
    - Red/pink  → words pushing toward FAKE
    - Green     → words pushing toward REAL
    - Opacity scales with influence strength.
    """
    parts = []
    for tok in annotated_tokens:
        word = tok["word"]
        direction = tok["direction"]
        score = tok["score"]

        if direction == "neutral" or score < 0.001:
            parts.append(f'<span style="color:#cbd5e1">{word}</span>')
        elif direction == "fake":
            # Red highlight — intensity proportional to score
            alpha = min(0.85, 0.25 + score * 2.5)
            parts.append(
                f'<span style="background:rgba(239,68,68,{alpha:.2f});'
                f'color:#fff;border-radius:3px;padding:1px 3px;'
                f'font-weight:600" title="→ FAKE ({score:.3f})">{word}</span>'
            )
        else:
            # Green highlight — pushes toward REAL
            alpha = min(0.85, 0.25 + score * 2.5)
            parts.append(
                f'<span style="background:rgba(34,197,94,{alpha:.2f});'
                f'color:#fff;border-radius:3px;padding:1px 3px;'
                f'font-weight:600" title="→ REAL ({score:.3f})">{word}</span>'
            )
    return "".join(parts)
