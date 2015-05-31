#!/bin/zsh

# Not really in a working state right now (?)

# Usage: ./setup.sh

set -e

echo "Make sure that conda (miniconda) is installed before trying to set up" \
     "or else this script will fail...\n"

echo "Creating conda environment...\n"
conda config --add channels pypi
conda create --yes -n reviews python=3.4
source activate reviews
conda install --yes --file conda_requirements.txt || \
    (echo "Wouldn't fucking work. Exiting."; exit 1)
echo "Created \"reviews\" environment successfully! To use environment, run" \
     "\"source activate reviews\". To get out of the environment, run" \
     "\"source deactivate\"."

echo "Installing some extra packages with pip (since conda does not seem to" \
     "want to install them)..."
pip install skll langdetect argparse

# Download model data for spaCy
echo "Downloading model data for spaCy package...\n"
python3.4 -m spacy.en.download

# Cythonize .pyx modules..."
UTIL_DIR=$(dirname $(readlink -f $0))
${UTIL_DIR}/cythonize.sh

echo "Complete."