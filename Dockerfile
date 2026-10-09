# Container for public cloud deployment (Render, Railway, Docker...).
# Step-by-step guide: docs/DEPLOY_RENDER.md
#
# Build & run locally:
#   docker build -t churn-dashboard .
#   docker run -p 7860:7860 churn-dashboard      -> http://localhost:7860
FROM python:3.12-slim

# Cloud hosts run containers securely as a non-root user with UID 1000, so the
# app (and its logs/ folder) must be owned by that user or logging would fail.
RUN useradd --create-home --uid 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=7860
USER user
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image ~1 GB smaller than the default CUDA build.
RUN pip install --user --index-url https://download.pytorch.org/whl/cpu torch
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# Only what the API needs at runtime: code, trained artefacts and the demo CSV.
COPY --chown=user src/ src/
COPY --chown=user app/ app/
COPY --chown=user models/ models/
COPY --chown=user data/sample_demo.csv data/sample_demo.csv

# WORKDIR is created by root, so hand the app folder (and a logs/ folder) to the
# runtime user — otherwise the prediction logger cannot create or write its files.
USER root
RUN mkdir -p logs && chown -R user:user /home/user/app
USER user

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
