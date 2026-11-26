# Financial Document Sentiment Analysis

A binary NLP classification project that identifies whether a financial news sentence or document excerpt expresses **positive** or **negative** sentiment. It uses **TF-IDF + Logistic Regression**, evaluates the model with **ROC-AUC**, and exposes trained predictions through a small **FastAPI** service.

> **Dataset note:** This repository contains code only. The Financial PhraseBank dataset is not redistributed here. Follow the instructions in [data/README.md](data/README.md) and the dataset's applicable terms before using it.

## Project highlights

- Binary classification: positive = 1, negative = 0.
- Neutral samples are intentionally excluded rather than merged into either class.
- TF-IDF vectorization with unigrams and bigrams.
- Logistic Regression with class balancing.
- A stratified hold-out test split and fixed random seed.
- Accuracy, precision, recall, F1-score, classification report, and **ROC-AUC**.
- ROC curve and confusion-matrix visualizations.
- FastAPI endpoints for health checks and text predictions.
- Unit tests for dataset parsing, prediction behaviour, and the API.

## Architecture

```text
Financial PhraseBank
       |
       v
Dataset loader (positive / negative only)
       |
       v
Stratified train / test split
       |
       v
TF-IDF vectorizer -> Logistic Regression
       |
       +----> Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
       |
       +----> ROC curve + Confusion matrix
       |
       +----> Saved scikit-learn Pipeline
                       |
                       v
                 FastAPI /predict
```

## Repository structure

```text
.
├── data/
│   └── README.md
├── models/                 # generated locally; not committed
├── reports/                # generated locally; not committed
├── src/
│   ├── api.py
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_predict.py
├── .gitignore
└── requirements.txt
```

## 1. Set up the environment

Python 3.10+ is recommended.

```bash
git clone https://github.com/Tejas4269/financial-document-sentiment-analysis.git
cd financial-document-sentiment-analysis

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Add the dataset

Download/obtain the Financial PhraseBank dataset under its applicable terms. Place an agreement-level file such as `Sentences_75Agree.txt` here:

```text
data/raw/Sentences_75Agree.txt
```

More detail: [data/README.md](data/README.md).

## 3. Train and evaluate

```bash
python -m src.train
```

To select a different dataset file or test split:

```bash
python -m src.train --data data/raw/Sentences_75Agree.txt --test-size 0.2
```

The script prints the measured test-set metrics, including:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
```

It also generates:

- `reports/metrics.json`
- `reports/roc_curve.png`
- `reports/confusion_matrix.png`
- `models/financial_sentiment_pipeline.joblib`

**ROC-AUC is computed from predicted positive-class probabilities on the held-out test data.** No score is hardcoded in this repository; the reported value is produced by running the training script against the actual dataset. This makes the result reproducible and avoids presenting an unmeasured number.

## 4. Run the FastAPI service

Train the model first, then start the API:

```bash
uvicorn src.api:app --reload
```

Open the interactive API documentation at http://127.0.0.1:8000/docs.

### Health check

`GET /health`

Example response:

```json
{"status": "ok"}
```

### Predict sentiment

`POST /predict`

Request:

```json
{
  "text": "The company reported stronger revenue and improved operating margins."
}
```

Example response (illustrative only; actual probabilities depend on the trained model):

```json
{
  "sentiment": "positive",
  "probability_positive": 0.87,
  "confidence": 0.87
}
```

## 5. Run tests

```bash
pytest -q
```

## Methodology and design decisions

1. **Binary target:** The source corpus has positive, neutral, and negative labels. Neutral records are omitted to keep the requested task genuinely binary.
2. **Pipeline:** TF-IDF and Logistic Regression are kept in one scikit-learn Pipeline. The vectorizer is fitted only on training text, preventing vocabulary leakage from the test set.
3. **Stratification:** The split preserves the class proportions in train and test partitions.
4. **ROC-AUC:** Uses the continuous positive-class probability, not the predicted class label.
5. **Reproducibility:** A fixed random seed is used for splitting and model configuration.
6. **Serving:** FastAPI validates input and returns the predicted class, positive-class probability, and confidence.

## Limitations

- Financial PhraseBank consists of financial news sentences, not full annual reports or arbitrary long documents. For long documents, split into meaningful sections/sentences and aggregate predictions carefully.
- Sentiment depends on context, domain terminology, and the dataset's annotation policy.
- A probability from Logistic Regression is not automatically a calibrated measure of real-world certainty.
- This is a demonstration/portfolio project, not investment advice or a production financial decision system.
- Evaluation scores depend on the selected PhraseBank agreement subset and train/test split. Always report the score generated from your own run.

## References

- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts*. Journal of the Association for Information Science and Technology.
- Financial PhraseBank dataset details and access information: see [data/README.md](data/README.md).

---

Built as a practical NLP + machine learning + FastAPI portfolio project.
