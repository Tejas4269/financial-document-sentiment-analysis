"""FastAPI inference service for the trained sentiment model."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_sentiment

app = FastAPI(
    title="Financial Document Sentiment API",
    description="Binary positive/negative sentiment classification for financial text.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000, description="Financial text")


class PredictionResponse(BaseModel):
    sentiment: str
    probability_positive: float
    confidence: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    try:
        return predict_sentiment(request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
