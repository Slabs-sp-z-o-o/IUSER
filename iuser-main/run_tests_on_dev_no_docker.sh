#!/bin/bash -e

cd "$(dirname "$0")"

source common/.env_development
export $(cut -d= -f1 common/.env_development)

pytest -ra \
  --log-level=info \
  --log-cli-level=info \
  -p no:cacheprovider \
  -m "not behave_setup" \
  -s \
  "$@"
