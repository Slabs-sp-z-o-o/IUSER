#!/bin/bash -e

cd "$(dirname "$0")" || exit

echo "Deploying REST API to GCP using Cloud Run. Project name: ${PROJECT_ID}"

# check if $PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
  echo "PROJECT_ID is not set. Please set it in the environment."
  exit 1
fi

read -p "Press enter to continue..."

. common/scripts/gcp.sh

ENV_FILENAME=$(get_env_filename)
source ./common/$ENV_FILENAME
export $(cut -d= -f1 ./common/$ENV_FILENAME)

SQLALCHEMY_DATABASE_URI=$(get_sqlalchemy_database_uri)

gcloud config set project $PROJECT_ID
gcloud builds submit --config cloud_build_rest_api.yaml
gcloud run deploy "$NODES_API_SERVICE_NAME" \
  --image gcr.io/$PROJECT_ID/rest_api \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 5000 \
  --set-env-vars SQLALCHEMY_DATABASE_URI="$SQLALCHEMY_DATABASE_URI" \
  --set-env-vars NODES_API_USERNAME="$NODES_API_USERNAME" \
  --set-env-vars NODES_API_PASSWORD="$NODES_API_PASSWORD"

echo "Be careful! Temporarily updating traffic to serve the latest revision!"

gcloud run services update-traffic "$NODES_API_SERVICE_NAME" --to-latest \
  --platform managed \
  --region $REGION
