#!/bin/zsh

# Usage: ./setup.sh

cd $(dirname $(dirname $0))
conda config --add channels pypi
conda config --add channels dan_blanchard
conda create --yes -n reviews python=3.4
source activate reviews
conda install --yes --file conda_requirements.txt || (echo "Wouldn't fucking work. Exiting."; exit 1)
pip install langdetect argparse