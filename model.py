"""
model.py — Training, saving, loading, and prediction for the Fake News Detector.

Pipeline:
  TfidfVectorizer (character + word n-grams)
  → PassiveAggressiveClassifier (online learning, efficient for text)

The trained pipeline is persisted to disk as a pickle file so it only
needs to be trained once.
"""

import os
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from data_loader import load_dataset, clean_text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")

LABEL_MAP = {0: "REAL", 1: "FAKE"}
LABEL_COLORS = {"REAL": "#22c55e", "FAKE": "#ef4444"}


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Build the sklearn Pipeline:
      - TF-IDF with word uni/bigrams + char ngrams for robustness
      - PassiveAggressiveClassifier: fast, online, great for text classification
    """
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),          # unigrams + bigrams
        max_features=60_000,
        sublinear_tf=True,           # apply log(TF) scaling
        min_df=1,                    # keep rare terms (important for small datasets)
        strip_accents="unicode",
        token_pattern=r"\b[a-z']{2,}\b",
    )
    # SGDClassifier with PA1 loss — identical to PassiveAggressiveClassifier
    # but uses the modern, non-deprecated API introduced in scikit-learn 1.8
    clf = SGDClassifier(
        loss="hinge",           # PA-style hinge loss
        penalty=None,           # no L2 regularization (pure PA)
        learning_rate="pa1",    # Passive Aggressive update rule
        eta0=0.5,               # equivalent to C parameter in PAC
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data_source: str = "sample", save: bool = True) -> dict:
    """
    Train the pipeline on the specified data source.

    Args:
        data_source: passed directly to data_loader.load_dataset()
        save:        whether to persist the trained model to disk

    Returns:
        dict with keys: pipeline, accuracy, report, train_size, test_size
    """
    print(f"[model] Loading dataset from: {data_source}")
    df = load_dataset(data_source)
    print(f"[model] Dataset shape: {df.shape} | Label distribution:\n{df['label'].value_counts()}")

    X = df["text"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    print(f"[model] Training on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["REAL", "FAKE"])

    print(f"[model] Test accuracy: {acc:.4f}")
    print(report)

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[model] Model saved → {MODEL_PATH}")

    return {
        "pipeline": pipeline,
        "accuracy": acc,
        "report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model() -> Pipeline:
    """
    Load the trained pipeline from disk.
    Raises FileNotFoundError if model has not been trained yet.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python model.py` to train and save the model first."
        )
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def get_or_train_model(data_source: str = "sample") -> Pipeline:
    """
    Return the trained model pipeline.
    Trains and saves the model automatically if it doesn't exist yet.
    This is the recommended entry point for the app.
    """
    if os.path.exists(MODEL_PATH):
        print(f"[model] Loading existing model from {MODEL_PATH}")
        return load_model()
    else:
        print("[model] No saved model found — training now...")
        result = train(data_source=data_source, save=True)
        return result["pipeline"]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(pipeline: Pipeline, text: str) -> dict:
    """
    Predict whether a news article is Real or Fake.

    Args:
        pipeline: trained sklearn Pipeline
        text:     raw news text (will be cleaned internally)

    Returns:
        dict with:
          label        — 'REAL' or 'FAKE'
          label_int    — 0 (real) or 1 (fake)
          confidence   — probability of the predicted class (0–100 %)
          probabilities — dict {'REAL': float, 'FAKE': float}
    """
    cleaned = clean_text(text)
    if not cleaned.strip():
        return {
            "label": "UNKNOWN",
            "label_int": -1,
            "confidence": 0.0,
            "probabilities": {"REAL": 0.0, "FAKE": 0.0},
        }

    label_int = int(pipeline.predict([cleaned])[0])

    # PassiveAggressiveClassifier doesn't natively support predict_proba.
    # We derive a soft confidence score from the decision function.
    decision = pipeline.decision_function([cleaned])[0]
    # Sigmoid to convert distance to probability
    prob_fake = float(1 / (1 + np.exp(-decision)))
    prob_real = 1.0 - prob_fake

    # Align probabilities with actual prediction
    if label_int == 0:
        confidence = prob_real * 100
    else:
        confidence = prob_fake * 100

    return {
        "label": LABEL_MAP[label_int],
        "label_int": label_int,
        "confidence": round(confidence, 1),
        "probabilities": {
            "REAL": round(prob_real * 100, 1),
            "FAKE": round(prob_fake * 100, 1),
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point — python model.py → trains and evaluates the model
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Fake News Detection model")
    parser.add_argument(
        "--data", default="sample",
        help="Dataset source: 'sample' | path/to/file.csv | path/to/liar_dir"
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retrain even if a saved model exists"
    )
    args = parser.parse_args()

    if args.retrain and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print("[model] Existing model deleted for retraining.")

    results = train(data_source=args.data, save=True)
    print(f"\n✅ Training complete. Accuracy: {results['accuracy']:.4f}")
    print(f"   Train samples: {results['train_size']} | Test samples: {results['test_size']}")
