# One image runs locally, in compose, and on Cloud Run. The PDFs, the embedding
# model (./models) and the Qdrant index (./qdrant_data) are baked in at build time;
# ingest needs no API key. Runtime needs only ANTHROPIC_API_KEY (+ APP_PASSWORD).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE=/app/models \
    QDRANT_PATH=/app/qdrant_data \
    LOG_DIR=/app/logs \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python scripts/build_corpus.py && python -m app.ingest --reset

# Everything is baked in; never reach out to the model hub at runtime.
ENV HF_HUB_OFFLINE=1

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/logs \
    && chown -R app:app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python scripts/healthcheck.py

CMD exec streamlit run ui/streamlit_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true
