#!/bin/bash -e

if [[ -z "${LOCAL_ML}" ]]; then
  export LOCAL_ML=FALSE
fi


cd "$(dirname "$0")"

source common/.env_development
export $(cut -d= -f1 common/.env_development)

IMAGE_NAME=pwrability-main-pytest-tests

docker build -t ${IMAGE_NAME} . --target pytest_tests

echo "=================== INFO ============================"
echo "You are about to run tests on $PROJECT_ID environment !"
echo "Local Machine Learning backend flag is set to: ${LOCAL_ML}"
read -p "Press enter to continue..."

. common/scripts/gcp.sh

assert_google_credentials
SQLALCHEMY_DATABASE_URI=$(get_sqlalchemy_database_uri)

docker run \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -it \
  -e LOCAL_ML \
  -v "$GOOGLE_CREDENTIALS_PATH":/app/serviceAccountKey.json \
  -v "$PWD"/tests/cloud/temporary_data:/app/tests/cloud/temporary_data \
  -v "$PWD"/tests/end_to_ends/temporary_data:/app/tests/end_to_ends/temporary_data \
  -v "$PWD"/tests/end_to_ends/tests_results:/app/tests/end_to_ends/tests_results \
  --env GOOGLE_APPLICATION_CREDENTIALS="/app/serviceAccountKey.json" \
  --env SQLALCHEMY_DATABASE_URI="$SQLALCHEMY_DATABASE_URI" \
  --network=host \
  ${IMAGE_NAME} \
  ./run_tests_on_dev_no_docker.sh "$@"

docker run \
    -v ${PWD}:/app \
    ${IMAGE_NAME} \
    chown -R $(id -u):$(id -g) /app
