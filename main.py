import os
from functools import lru_cache
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# NLP
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Topic modeling / clustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import numpy as np

# --- FastAPI setup ---
app = FastAPI(title="Content Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models / Pipelines ---

def _ensure_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

@lru_cache(maxsize=1)
def get_vader():
    _ensure_vader()
    return SentimentIntensityAnalyzer()

@lru_cache(maxsize=1)
def get_hf_sentiment():
    # RoBERTa-based fine-tuned on sentiment
    return pipeline("sentiment-analysis")

@lru_cache(maxsize=1)
def get_emotion_pipeline():
    # DistilRoBERTa emotion model
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        truncation=True,
    )

@lru_cache(maxsize=1)
def get_toxicity_pipeline():
    # Binary toxicity classifier; probabilities for toxic vs non-toxic
    return pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        return_all_scores=True,
        truncation=True,
    )

@lru_cache(maxsize=1)
def get_zero_shot():
    # Used for tone & type classification
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# --- Schemas ---
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze")

class TextBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)

class SentimentResponse(BaseModel):
    compound: float
    positive: float
    negative: float
    neutral: float
    label: Literal["Positive", "Negative", "Neutral", "Mixed"]

class EmotionResponse(BaseModel):
    scores: dict
    dominant: str

class TopicRequest(BaseModel):
    texts: List[str] = Field(..., min_items=2, description="Corpus for topic discovery (>=2)")
    method: Literal["lda", "kmeans"] = "lda"
    num_topics: int = Field(5, ge=2, le=25)
    max_features: int = Field(5000, ge=500, le=20000)
    n_words: int = Field(8, ge=3, le=30, description="Top words per topic")

class TopicResponse(BaseModel):
    topics: List[dict]

class ModerationResponse(BaseModel):
    toxic: bool
    toxicity_score: float
    categories: List[str]

class ClassifyResponse(BaseModel):
    tone: str
    length: Literal["short", "medium", "long"]
    type: str

# --- Helpers ---

def label_from_vader_scores(scores: dict) -> str:
    comp = scores["compound"]
    pos = scores["pos"]
    neg = scores["neg"]
    neu = scores["neu"]
    if comp >= 0.4 and pos > neg:
        return "Positive"
    if comp <= -0.4 and neg > pos:
        return "Negative"
    if max(pos, neg) < 0.3:
        return "Neutral"
    return "Mixed"

# --- Core Logic ---

def _get_sentiment(text: str, engine: Literal["vader", "transformer"]) -> SentimentResponse:
    """Performs sentiment analysis on a given text."""
    if engine == "vader":
        sia = get_vader()
        s = sia.polarity_scores(text)
        return SentimentResponse(
            compound=s["compound"], positive=s["pos"], negative=s["neg"], neutral=s["neu"], label=label_from_vader_scores(s)
        )

    # transformer engine
    clf = get_hf_sentiment()
    out = clf(text)[0]
    # Normalize to a VADER-like response
    label = out["label"].upper()
    score = float(out["score"])  # confidence of predicted label
    # Map to pos/neg/neu distributions
    pos = score if "POS" in label else 0.0
    neg = score if "NEG" in label else 0.0
    neu = 1.0 - score if (pos > 0 or neg > 0) else 1.0
    compound = (pos - neg)
    return SentimentResponse(
        compound=compound,
        positive=pos,
        negative=neg,
        neutral=neu,
        label="Positive" if pos > 0.6 else ("Negative" if neg > 0.6 else ("Neutral" if neu > 0.6 else "Mixed")),
    )

def _get_emotion(text: str) -> EmotionResponse:
    """Performs emotion analysis on a given text."""
    emo = get_emotion_pipeline()(text)[0]
    scores = {e["label"].lower(): float(e["score"]) for e in emo}
    dominant = max(scores, key=scores.get)
    return EmotionResponse(scores=scores, dominant=dominant)

def _get_moderation(text: str, nsfw_labels: Optional[List[str]] = None) -> ModerationResponse:
    """Performs moderation analysis on a given text."""
    tox = get_toxicity_pipeline()(text)[0]
    scores = {item["label"].lower(): float(item["score"]) for item in tox}
    toxicity_score = scores.get("toxic", 0.0)
    toxic = toxicity_score >= 0.5
    categories = ["toxic"] if toxic else []

    if nsfw_labels:
        zsc = get_zero_shot()
        z = zsc(text, candidate_labels=nsfw_labels, multi_label=True)
        flagged = [lbl for lbl, sc in zip(z["labels"], z["scores"]) if sc >= 0.5]
        categories.extend(flagged)

    return ModerationResponse(toxic=toxic, toxicity_score=float(toxicity_score), categories=categories)

def _get_classification(text: str) -> ClassifyResponse:
    """Performs content classification (tone, length, type) on a given text."""
    wc = len(text.split())
    length = "short" if wc <= 20 else ("medium" if wc <= 100 else "long")
    zsc = get_zero_shot()
    tone = zsc(text, candidate_labels=["formal", "informal", "neutral", "sarcastic", "polite", "angry"])["labels"][0]
    type_label = zsc(text, candidate_labels=["rant", "advice", "question", "review", "story", "announcement"])["labels"][0]
    return ClassifyResponse(tone=tone, length=length, type=type_label)

# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/content/sentiment", response_model=SentimentResponse)
def sentiment(req: TextRequest, engine: Literal["vader", "transformer"] = Query("vader")):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Empty text")

    return _get_sentiment(text, engine)

@app.post("/api/content/emotion", response_model=EmotionResponse)
def emotion(req: TextRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Empty text")
    return _get_emotion(text)


@app.post("/api/content/topics", response_model=TopicResponse)
def topics(req: TopicRequest):
    docs = [t.strip() for t in req.texts if t and t.strip()]
    if len(docs) < 2:
        raise HTTPException(400, "Provide at least 2 non-empty texts for topic modeling")

    if req.method == "lda":
        vectorizer = CountVectorizer(max_features=req.max_features, stop_words="english")
        X = vectorizer.fit_transform(docs)
        lda = LatentDirichletAllocation(n_components=req.num_topics, random_state=42)
        lda.fit(X)
        words = np.array(vectorizer.get_feature_names_out())
        topics = []
        for i, comps in enumerate(lda.components_):
            top_idx = comps.argsort()[-req.n_words:][::-1]
            topics.append({"topic": int(i), "keywords": words[top_idx].tolist()})
        return TopicResponse(topics=topics)

    # k-means clusters with TF-IDF
    vectorizer = TfidfVectorizer(max_features=req.max_features, stop_words="english")
    X = vectorizer.fit_transform(docs)
    km = KMeans(n_clusters=req.num_topics, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    words = np.array(vectorizer.get_feature_names_out())
    centers = km.cluster_centers_
    topics = []
    for i in range(req.num_topics):
        center = centers[i]
        top_idx = center.argsort()[-req.n_words:][::-1]
        topics.append({"topic": int(i), "keywords": words[top_idx].tolist()})
    return TopicResponse(topics=topics)

@app.post("/api/content/moderation", response_model=ModerationResponse)
def moderation(req: TextRequest, nsfw_labels: Optional[List[str]] = Query(None, description="Custom NSFW labels to flag via zero-shot, optional")):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Empty text")
    return _get_moderation(text, nsfw_labels)

@app.post("/api/content/classify", response_model=ClassifyResponse)
def classify(req: TextRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Empty text")
    return _get_classification(text)

# Optional: one-call batch analyzer for React convenience
class AnalyzeItem(BaseModel):
    id: Optional[str] = None
    text: str

class AnalyzeBatchRequest(BaseModel):
    items: List[AnalyzeItem]

class AnalyzeResult(BaseModel):
    id: Optional[str]
    sentiment: SentimentResponse
    emotion: EmotionResponse
    moderation: ModerationResponse
    classification: ClassifyResponse

class AnalyzeBatchResponse(BaseModel):
    results: List[AnalyzeResult]

@app.post("/api/content/analyze", response_model=AnalyzeBatchResponse)
def analyze(req: AnalyzeBatchRequest):
    results: List[AnalyzeResult] = []
    for item in req.items:
        s = _get_sentiment(item.text, "transformer") # Using the more powerful engine for batch
        e = _get_emotion(item.text)
        m = _get_moderation(item.text) # Can extend to pass nsfw_labels from request
        c = _get_classification(item.text)
        results.append(AnalyzeResult(id=item.id, sentiment=s, emotion=e, moderation=m, classification=c))
    return AnalyzeBatchResponse(results=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))