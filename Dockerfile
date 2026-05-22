# Python base image
FROM python:3.10-slim

# Prevent Python caching & buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    dos2unix \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first (cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy project
COPY . .

# FIX ENTRYPOINT (VERY IMPORTANT for Windows users)
RUN dos2unix entrypoint.sh || true && \
    chmod +x entrypoint.sh

# Ensure script exists (debug safety)
RUN ls -la /app

# Expose FastAPI port
EXPOSE 8000

# Use bash explicitly (more stable than ./entrypoint.sh)
CMD ["bash", "./entrypoint.sh"]