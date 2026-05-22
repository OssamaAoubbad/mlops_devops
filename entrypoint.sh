#!/bin/bash
# Stop execution immediately if any command fails
set -e

echo "========================================"
echo "1. Fetching Datasets from MinIO..."
echo "========================================"
# DVC will use the credentials injected by docker-compose to pull the CSVs
dvc pull

echo "========================================"
echo "2. Starting FastAPI & Ray Serve..."
echo "========================================"
# We use the $RUN_ID environment variable (which we will define in docker-compose)
# to tell the API exactly which trained model to load from MLflow.
exec python madewithml/serve.py --run_id "$RUN_ID"