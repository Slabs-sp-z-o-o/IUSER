#!/bin/bash -e

export LOCAL_ML=TRUE

./run_behave_tests_on_dev_in_docker.sh "$@"