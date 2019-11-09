#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/utils.sh

# Train tf 
print_header "Running full test suite"

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Experiment
cd $DIR
python src/run_full_test_suite.py