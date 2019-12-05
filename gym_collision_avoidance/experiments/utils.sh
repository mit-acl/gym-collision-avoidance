#!/bin/bash
set -e

function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Disable all tensorflow warnings/info (keep errors)
export TF_CPP_MIN_LOG_LEVEL=2

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR=${DIR}/../..
source $BASE_DIR/venv/bin/activate
export PYTHONPATH=${BASE_DIR}/venv/bin/python/dist-packages
echo "Entered virtualenv."