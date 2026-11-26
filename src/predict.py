"""Load the trained model and classify financial text."""
from pathlib import Path
import joblib


DEFAULT_MODEL = Path("models/financial_sentiment_pipeline.joblib")


def predict_sentiment(text: str, model_path: str | Path = DEFAULT_MODEL) -> dict:
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. Run 'python -m src.train' first."
        )

    model = joblib.load(path)
    probability_positive = float(model.predict_proba([text])[0, 1])
    label = "positive" if probability_positive >= 0.5 else "negative"
    confidence = (
        probability_positive if label == "positive" else 1.0 - probability_positive
    )
    return {
        "sentiment": label,
        "probability_positive": round(probability_positive, 6),
        "confidence": round(confidence, 6),
    }
