import json
import time
from http import HTTPStatus
from typing import Any

import ray
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel
from ray import serve

from madewithml import predict
from madewithml.config import MLFLOW_TRACKING_URI, logger, mlflow, settings

# ==========================================
# 1. Prometheus Metrics Definition
# ==========================================

# --- System & API Metrics ---
REQUEST_COUNT = Counter(
    "api_requests_total", 
    "Total number of requests received by the API", 
    ["method", "endpoint"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", 
    "Latency of API requests in seconds", 
    ["endpoint"]
)

# --- Data Drift Metrics ---
INPUT_LENGTH = Histogram(
    "model_input_word_count",
    "Number of words in the combined input text",
    buckets=[5, 10, 20, 50, 100, 200, 500]
)

# --- Model Confidence & Concept Drift Metrics ---
PREDICTION_COUNT = Counter(
    "model_predictions_total", 
    "Total number of predictions made, grouped by predicted class", 
    ["predicted_class"]
)
PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Probability score of the chosen class",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
)
FALLBACK_COUNT = Counter(
    "model_fallback_total",
    "Number of times the model fell back to the 'other' class due to low confidence"
)

# --- Performance Feedback Metrics ---
CORRECT_PREDICTIONS = Counter(
    "model_correct_predictions_total",
    "Count of correct predictions based on user feedback"
)
INCORRECT_PREDICTIONS = Counter(
    "model_incorrect_predictions_total",
    "Count of incorrect predictions based on user feedback"
)

# ==========================================
# 2. FastAPI Application Setup
# ==========================================
app = FastAPI(
    title="Made With ML - Production API",
    description="Classify machine learning projects with built-in MLOps observability.",
    version="1.0",
)

# Mount the Prometheus ASGI app. Grafana will hit http://<host>:8000/metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Pydantic schemas for strict payload validation
class PredictPayload(BaseModel):
    title: str
    description: str

class FeedbackPayload(BaseModel):
    prediction_id: str
    predicted_tag: str
    true_tag: str

# ==========================================
# 3. Ray Serve Deployment
# ==========================================
@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, run_id: str, threshold: float = 0.9):
        """Initialize the model and load weights into memory."""
        self.run_id = run_id
        self.threshold = threshold
        
        # Ensure workers have access to the model registry via EFS/Docker Volume
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
        self.predictor = predict.TorchPredictor.from_checkpoint(best_checkpoint)
        
        logger.info(f"Model Deployment initialized with Run ID: {self.run_id}")

    @app.get("/")
    def _health_check(self) -> dict[str, Any]:
        """Health Check for Container Orchestration."""
        REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
        return {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "environment": settings.environment
        }

    @app.get("/metadata/")
    def _metadata(self) -> dict[str, str]:
        """Returns the currently active model version."""
        REQUEST_COUNT.labels(method="GET", endpoint="/metadata/").inc()
        return {"run_id": self.run_id, "threshold": str(self.threshold)}

    @app.post("/predict/")
    async def _predict(self, payload: PredictPayload, request: Request) -> dict[str, Any]:
        """Core inference endpoint with full observability tracking."""
        start_time = time.time()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/").inc()

        # 1. Track Input Data Metrics (Data Drift)
        combined_text = f"{payload.title} {payload.description}"
        word_count = len(combined_text.split())
        INPUT_LENGTH.observe(word_count)

        # 2. Package and run Inference
        sample_ds = ray.data.from_items([{
            "title": payload.title, 
            "description": payload.description, 
            "tag": "other"
        }])
        results = predict.predict_proba_dataset(ds=sample_ds, predictor=self.predictor)

        # 3. Apply business logic and track Model Metrics (Confidence Drift)
        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            max_prob = prob[pred]
            
            # Observe raw confidence before thresholding
            PREDICTION_CONFIDENCE.observe(max_prob)
            
            # Fallback logic
            if max_prob < self.threshold:
                results[i]["prediction"] = "other"
                pred = "other"
                FALLBACK_COUNT.inc()
                
            # Update prediction distribution metric
            PREDICTION_COUNT.labels(predicted_class=pred).inc()

        # 4. Track Latency
        process_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict/").observe(process_time)

        # 5. Log raw inputs for asynchronous drift analysis systems (e.g., Evidently)
        logger.info(json.dumps({
            "event": "inference",
            "input_title": payload.title,
            "input_description": payload.description,
            "output_prediction": results[0]["prediction"],
            "output_probabilities": results[0]["probabilities"]
        }))

        return {"results": results, "latency_seconds": process_time}

    @app.post("/feedback/")
    async def _log_feedback(self, payload: FeedbackPayload) -> dict[str, Any]:
        """Receives ground truth data to calculate real-time model accuracy."""
        REQUEST_COUNT.labels(method="POST", endpoint="/feedback/").inc()
        
        # Calculate real-time accuracy statelessly
        if payload.predicted_tag == payload.true_tag:
            CORRECT_PREDICTIONS.inc()
        else:
            INCORRECT_PREDICTIONS.inc()
        
        # Log for historical tracking
        logger.info(json.dumps({
            "event": "ground_truth_feedback",
            "prediction_id": payload.prediction_id,
            "predicted_tag": payload.predicted_tag,
            "true_tag": payload.true_tag
        }))
        
        return {"status": "Feedback logged successfully and metrics updated."}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.", required=True)
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for `other` class.")
    args = parser.parse_args()
    
    # Initialize Ray and deploy the FastAPI app
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    serve.run(ModelDeployment.bind(run_id=args.run_id, threshold=args.threshold))