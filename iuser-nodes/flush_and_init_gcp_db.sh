#!/bin/bash -e

cd "$(dirname "$0")"

echo "=================== DANGER!!! ============================"
echo "This script will flush all the data from Cloud SQL of project of id: $PROJECT_ID !"
echo "Use it at your own risk."
read -p "Press enter to flush the database..."

. common/scripts/gcp.sh
export SQLALCHEMY_DATABASE_URI=$(get_sqlalchemy_database_uri)

read -p "Add examples to the database (y/n)?" CONT
if [ "$CONT" = "y" ]; then
  export CREATE_DB_ADD_EXAMPLES=True
fi

docker build --tag gcp_init_db --target init_db .
docker run --env SQLALCHEMY_DATABASE_URI --env CREATE_DB_ADD_EXAMPLES --env NODES_API_USERNAME --env NODES_API_PASSWORD gcp_init_db
