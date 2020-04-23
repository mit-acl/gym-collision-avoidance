#!/bin/bash
set -e

MAKE_VENV=${1:-true}
SOURCE_VENV=${2:-true}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if $MAKE_VENV; then
    # Virtualenv w/ python3
    export PYTHONPATH=/usr/bin/python3 # point to your python3
    python3 -m pip install virtualenv
    cd $DIR
    python3 -m virtualenv venv
fi

if $SOURCE_VENV; then
    cd $DIR
    source venv/bin/activate
    export PYTHONPATH=${DIR}/venv/bin/python/dist-packages
fi

# Install this pkg and its requirements
python -m pip install -e $DIR
python -m pip install git+https://github.com/openai/baselines.git@ea25b9e8b234e6ee1bca43083f8f3cf974143998

# Install RVO and its requirements
cd $DIR/gym_collision_avoidance/envs/policies/Python-RVO2
python -m pip install Cython
if [[ "$OSTYPE" == "darwin"* ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.15
    brew install cmake
fi
python setup.py build
python setup.py install

# Install DRL Long's requirements
# python -m pip install torch torchvision

echo "Finished installing gym_collision_avoidance!"