#!/bin/bash
set -e

function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR=${DIR}/../..

# Virtualenv
python3 -m pip install virtualenv
cd $BASE_DIR
virtualenv venv
source venv/bin/activate

# Install this pkg and its requirements
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

export TF_CPP_MIN_LOG_LEVEL=2 
