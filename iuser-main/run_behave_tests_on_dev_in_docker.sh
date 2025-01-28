#!/bin/bash -e

if [[ -z "${LOCAL_ML}" ]]; then
  export LOCAL_ML=FALSE
fi

cd "$(dirname "$0")"

IMAGE_NAME=pwrability-main-behave-tests

. common/scripts/gcp.sh

source common/.env_development
export $(cut -d= -f1 common/.env_development)

assert_google_credentials
SQLALCHEMY_DATABASE_URI=$(get_sqlalchemy_database_uri)


docker build -t ${IMAGE_NAME}  . --target behave_tests

echo "=================== INFO ============================"
echo "You are about to run behave tests on $PROJECT_ID environment !"
echo "Local Machine Learning backend flag is set to: ${LOCAL_ML}"
read -p "Press enter to continue..."

# Data and nodes setup for behave
docker run \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -it \
  -e LOCAL_ML \
  -v "$GOOGLE_CREDENTIALS_PATH":/app/serviceAccountKey.json \
  --env GOOGLE_APPLICATION_CREDENTIALS="/app/serviceAccountKey.json" \
  --env SQLALCHEMY_DATABASE_URI="$SQLALCHEMY_DATABASE_URI" \
  ${IMAGE_NAME} \
  ../../run_tests_on_dev_no_docker.sh -m "behave_setup" tests/behave


docker run \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e LOCAL_ML \
  -it \
  -v "$GOOGLE_CREDENTIALS_PATH":/app/serviceAccountKey.json \
  -v "$PWD"/tests/behave:/app/tests/behave \
  --env GOOGLE_APPLICATION_CREDENTIALS="/app/serviceAccountKey.json" \
  --env-file common/.env_development \
  --network=host \
  ${IMAGE_NAME} \
  behave --logging-level=debug --no-logcapture ./features --tags="-failing" "$@"
