#!/bin/zsh

# Works on some types of systems, but maybe not others
# Note: having conda installed is a requirement (currently)

# Usage: ./setup.sh

set -e

echo "Make sure that conda (miniconda) is installed before trying to set up" \
     "or else this script will fail...\n"

echo "Creating conda environment...\n"
conda config --add channels pypi
# Create environment first and force python3.4 (for some reason, just adding
# python=3.4 to the list of packages in conda_requirements.txt does not work
# as it is not recognized as a valid package name.
conda create --yes -n reviews python=3.4
# And now install all of the packages we need
source activate reviews
conda install --yes --file conda_requirements.txt
if [[ $? -gt 0 ]]; then
    
    echo "Wouldn't work. Exiting.\n"
    exit 1
    
fi
echo "Created \"reviews\" environment successfully! To use environment, run" \
     "\"source activate reviews\". To get out of the environment, run" \
     "\"source deactivate\".\n"

echo "Installing some extra packages with pip (since conda does not seem to" \
     "want to install them)...\n"
pip install skll==1.1.0 langdetect argparse pudb
if [[ $? -gt 0 ]]; then
    
    echo "pip installation of langdetect and argparse failed. Exiting.\n"
    exit 1
    
fi

# Download model data for spaCy
echo "Downloading model data for spaCy package...\n"
python3.4 -m spacy.en.download

# Compile Cython modules
echo "Compiling Cython extensions...\n"
THIS_DIR=$(dirname $(readlink -f $0))
python3.4 ${THIS_DIR}/setup.py install
echo "Package installed!"
echo "If changes are made to the Cython extensions, run the following to " \
     "re-compile the extensions for use in the various command-line " \
     "utilities:\n\tpython setup.py build_ext\n"

echo "Setup complete. Use \"source activate reviews\" to activate conda" \
     "environment.\n"
