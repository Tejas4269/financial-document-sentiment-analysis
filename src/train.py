"""Train, evaluate, and persist the binary financial sentiment model."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

from src.data import load_phrasebank
from src.evaluate import evaluate_model
from src.model import build_model


def train(dataset_path: str, test_size: float = 0.2) -> dict:
    data = load_phrasebank(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=test_size,
        random_state=42,
        stratify=data["label"],
    )

    model = build_model()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "financial_sentiment_pipeline.joblib")

    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("Saved model to models/financial_sentiment_pipeline.joblib")
    print("Saved metrics and plots to reports/")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/raw/Sentences_75Agree.txt",
        help="Path to a Financial PhraseBank sentence file.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    train(args.data, args.test_size)


if __name__ == "__main__":
    main()
