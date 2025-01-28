#!/usr/bin/env bash

cd "$(dirname "$0")"/.. || exit

echo preparing tools...
docker build --quiet=true --tag=flake8 --target=flake8 utils/docker_flake8
docker build --quiet=true --tag=autopep8 --target=autopep8 utils/docker_flake8
docker build --quiet=true --tag=unify --target=unify utils/docker_flake8
docker build --quiet=true --tag=isort --target=isort utils/docker_flake8
RUN="docker run --rm --volume "$PWD:/prj" --user $EUID:$GROUPS"

echo fixing imports...
$RUN isort --jobs=4 .

echo fixing quotes...
$RUN unify --in-place --recursive .

echo fixing formatting...
$RUN autopep8 --in-place --recursive --jobs=-1 .

echo
echo remaining problems:
$RUN flake8 --statistics --count -q 2>/dev/null | grep -v ^\./

echo
echo changes summary:
git --no-pager diff --stat --compact-summary
