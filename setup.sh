#!/bin/bash

# Works on some types of systems, but maybe not others
# Note: having conda installed is a requirement (currently)

# Usage: ./setup.sh

set -eu

export PATH=$PATH:/opt/python/conda_default/bin
export CONDA_OLD_PS1="something" # Hack
ORIG_DIR=$(pwd)
THIS_DIR=$(dirname $(readlink -f $0))
cd ${THIS_DIR}

echo "Make sure that conda (miniconda) is installed before trying to set up" \
     "or else this script will fail..."

echo "Creating conda environment..."
conda config --add channels pypi
# Create environment first and force python=3.4 (for some reason, just
# adding python=3.4 to the list of packages in conda_requirements.txt
# does not work as it is not recognized as a valid package name)
conda create --yes -n reviews python=3.4
# And now install all of the packages we need
source activate reviews
conda install --yes --file conda_requirements.txt
if [ $? -gt 0 ]; then
    echo "\"conda install --yes --file conda_requirements.txt\" failed. " \
         "Exiting."
    cd ${ORIG_DIR}
    exit 1
fi
echo "Created \"reviews\" environment successfully! To use environment, run" \
     "\"source activate reviews\". To get out of the environment, run" \
     "\"source deactivate\"."

echo "Installing some extra packages with pip (since conda does not seem to" \
     "want to install them)..."
pip install skll==1.1.0 langdetect==1.0.5 argparse pudb nose2==0.5.0 typing==3.5.0.1 schema==0.4.0
if [ $? -gt 0 ]; then
    echo "pip installation of langdetect and argparse failed. Exiting."
    cd ${ORIG_DIR}
    exit 1
fi

# Download model data for spaCy
echo "Downloading model data for spaCy package..."
python3.4 -m spacy.en.download

# Compile Cython modules
echo "Compiling Cython extensions..."
python3.4 setup.py install
echo "Package installed!"
echo "If changes are made to the Cython extensions, run the following to " \
     "re-compile the extensions for use in the various command-line " \
     "utilities: \"python setup.py build_ext\". Or run the following to " \
     "reinstall the entire package: \"python setup.py install\""

echo "Setup complete. Use \"source activate reviews\" to activate conda" \
     "environment."
