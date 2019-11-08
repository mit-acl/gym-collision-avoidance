#!/bin/bash
set -e

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Virtualenv
python3 -m pip install virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate

# Install this pkg and its requirements
python3 -m pip install -r requirements.txt
python3 -m pip install -e .