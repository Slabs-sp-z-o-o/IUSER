#!/bin/sh

docker-compose build
docker-compose run tests
docker-compose down --remove-orphans --volumes --timeout 0
