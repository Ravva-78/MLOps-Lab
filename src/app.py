import logging
import time
import joblib
import numpy as np
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query

from src.schemas import (
    RecommendResponse,
    RecommendationItem,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionResult,
    HealthResponse
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MovieLens Recommender API",
    version="1.0.0"
)

model = None
features = None
model_version = "MovieLensRecommender/1"


# 🔥 STARTUP LOADING (LAB REQUIREMENT)
@app.on_event("startup")
def load_model():
    global model, features

    model = joblib.load("models/model.pkl")
    features = joblib.load("models/rating_features.pkl")

    logger.info("Model and features loaded successfully")


# ✅ HEALTH ENDPOINT
@app.get("/health", response_model=HealthResponse)
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "service": "MovieLens API",
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version
    }


# ✅ RECOMMEND
@app.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: int = Query(..., gt=0), n: int = Query(5, ge=1, le=50)):

    start_time = time.time()

    try:
        if user_id not in features.user_ids:
            raise HTTPException(status_code=404, detail="User not found")

        user_idx = np.where(features.user_ids == user_id)[0][0]
        user_vector = features.ratings_matrix.iloc[user_idx].values

        predictions = []

        for idx, movie_id in enumerate(features.movie_ids):
            if user_vector[idx] == 0:
                rating = model.predict_rating(user_id, movie_id)
                predictions.append((movie_id, rating))

        predictions.sort(key=lambda x: x[1], reverse=True)

        top_n = predictions[:n]

        recommendations = [
            RecommendationItem(
                movie_id=int(mid),
                predicted_rating=float(r),
                rank=i + 1
            )
            for i, (mid, r) in enumerate(top_n)
        ]

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version
        }

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ✅ SIMILAR USERS
@app.get("/similar_users/{user_id}")
def similar_users(user_id: int, k: int = Query(5, ge=1, le=20)):

    if user_id not in features.user_ids:
        raise HTTPException(status_code=404, detail="User not found")

    similar = features.get_similar_users(user_id, n=k)

    return {
        "user_id": user_id,
        "similar_users": [
            {"user_id": int(uid), "similarity": float(sim)}
            for uid, sim in similar
        ]
    }


# ✅ BATCH PREDICT (FIXED PROPERLY)
@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):

    start_time = time.time()

    try:
        results = []

        for item in request.predictions:
            user_id = item.user_id
            movie_id = item.movie_id

            if user_id not in features.user_ids:
                continue

            if movie_id not in features.movie_ids:
                continue

            user_idx = np.where(features.user_ids == user_id)[0][0]
            user_vector = features.ratings_matrix.iloc[user_idx].values

            movie_idx = np.where(features.movie_ids == movie_id)[0][0]

            distances, indices = model.kneighbors(
                [user_vector],
                n_neighbors=min(5, len(features.ratings_matrix))
            )

            neighbor_ratings = features.ratings_matrix.iloc[
                indices[0], movie_idx
            ].values

            predicted_rating = float(np.mean(neighbor_ratings))

            results.append(
                PredictionResult(
                    user_id=user_id,
                    movie_id=movie_id,
                    predicted_rating=predicted_rating
                )
            )

        latency = (time.time() - start_time) * 1000

        return {
            "predictions": results,
            "count": len(results),
            "latency_ms": latency
        }

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))