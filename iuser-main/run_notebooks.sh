#!/bin/bash -e

cd "$(dirname "$0")"

if [[ -z "${PWRABILITY_DEV_CREDENTIALS_PATH}" ]]; then
  echo "Please set PWRABILITY_DEV_CREDENTIALS_PATH environment variable!"
  echo "Tip:"
  echo "export PWRABILITY_DEV_CREDENTIALS_PATH="
  exit
fi

if [[ -z "${PWRABILITY_PROD_CREDENTIALS_PATH}" ]]; then
  echo "Please set PWRABILITY_PROD_CREDENTIALS_PATH environment variable!"
  echo "Tip:"
  echo "export PWRABILITY_PROD_CREDENTIALS_PATH="
  exit
fi

export CREDENTIALS_DEV_CONTAINER_PATH="/app/serviceAccountKeyDev.json"
export CREDENTIALS_PROD_CONTAINER_PATH="/app/serviceAccountKeyProd.json"

IMAGE_TAG="pwrability_notebooks"
docker build -t $IMAGE_TAG -f Dockerfile.notebooks .
docker run --rm -p 8888:8888 \
          -v "${PWD}":/home/jovyan/work  \
          --env CREDENTIALS_DEV_CONTAINER_PATH=$CREDENTIALS_DEV_CONTAINER_PATH \
          --env CREDENTIALS_PROD_CONTAINER_PATH=$CREDENTIALS_PROD_CONTAINER_PATH \
          --env GOOGLE_APPLICATION_CREDENTIALS=$CREDENTIALS_DEV_CONTAINER_PATH \
          -v $PWRABILITY_DEV_CREDENTIALS_PATH:$CREDENTIALS_DEV_CONTAINER_PATH \
          -v $PWRABILITY_PROD_CREDENTIALS_PATH:$CREDENTIALS_PROD_CONTAINER_PATH \
          $IMAGE_TAG
