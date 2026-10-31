#!/usr/bin/env bash
# Deploy to Google Cloud Run. Usage: PROJECT_ID=my-proj [REGION=us-central1] ./scripts/deploy_cloudrun.sh
#
# One-time setup (secrets must exist before the first deploy):
#   printf '%s' "$ANTHROPIC_API_KEY" | gcloud secrets create ANTHROPIC_API_KEY --data-file=- --project "$PROJECT_ID"
#   printf '%s' "$APP_PASSWORD"      | gcloud secrets create APP_PASSWORD      --data-file=- --project "$PROJECT_ID"
#   and grant the Cloud Run runtime service account roles/secretmanager.secretAccessor.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-riverbend-assistant}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE}"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com --project "$PROJECT_ID"

gcloud artifacts repositories describe cloud-run-source-deploy \
  --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1 \
  || gcloud artifacts repositories create cloud-run-source-deploy \
       --repository-format=docker --location "$REGION" --project "$PROJECT_ID"

gcloud builds submit --tag "$IMAGE" --project "$PROJECT_ID" .

# max-instances 1: embedded Qdrant is single-writer, and it caps cost.
# --allow-unauthenticated: the app password is the gate.
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --project "$PROJECT_ID" --region "$REGION" \
  --port 8080 --memory 2Gi --cpu 1 \
  --min-instances 0 --max-instances 1 \
  --timeout 3600 --session-affinity \
  --allow-unauthenticated \
  --set-secrets "ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,APP_PASSWORD=APP_PASSWORD:latest"

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format 'value(status.url)')"
echo "Deployed: $URL"
echo "Smoke test: python scripts/smoke_test.py $URL"
