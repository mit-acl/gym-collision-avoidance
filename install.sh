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

# Install RVO and its requirements
cd gym_collision_avoidance/envs/policies/Python-RVO2
python3 -m pip install Cython
if [[ "$OSTYPE" == "darwin"* ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.15
    brew install cmake
fi
python3 setup.py build
python3 setup.py install

# Install DRL Long's requirements
python3 -m pip install torch torchvision
