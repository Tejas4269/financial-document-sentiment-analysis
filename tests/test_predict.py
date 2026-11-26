import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from src.predict import predict_sentiment


def test_predict_returns_binary_sentiment_and_probabilities(tmp_path):
    model = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )
    model.fit(["growth outlook", "loss decline"], [1, 0])
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    result = predict_sentiment("growth outlook", model_path)

    assert result["sentiment"] in {"positive", "negative"}
    assert 0.0 <= result["probability_positive"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_empty_text_is_rejected():
    with pytest.raises(ValueError):
        predict_sentiment("   ")
