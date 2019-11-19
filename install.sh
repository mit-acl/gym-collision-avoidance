#!/bin/bash
set -e

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Virtualenv w/ python3
python3 -m pip install virtualenv
cd $DIR
virtualenv -p python3 venv
source venv/bin/activate
export PYTHONPATH=${DIR}/venv/bin/python/dist-packages

# Install this pkg and its requirements
python -m pip install -r requirements.txt
python -m pip install -e .

# Install RVO and its requirements
cd gym_collision_avoidance/envs/policies/Python-RVO2
python -m pip install Cython
if [[ "$OSTYPE" == "darwin"* ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.15
    brew install cmake
fi
python setup.py build
python setup.py install

# Install DRL Long's requirements
python -m pip install torch torchvision
