#!/bin/zsh

# Compile Python modules

source /home/research/mmulholland/.zshrc
source activate reviews

REVIEWS=$TEXTDYNAMIC/montclair/reviewer_experience_prediction
cd ${REVIEWS}

echo "Running cython on feature_extraction.pyx and then compiling with" \
    "gcc..."
cd src
cython -a feature_extraction.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I/opt/python/conda_default/pkgs/python-3.4.3-0/include/python3.4m -o \
    feature_extraction.so feature_extraction.c

echo "Running cython on mongodb.pyx and then compiling with gcc..."
cd ../util
cython -a mongodb.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I/opt/python/conda_default/pkgs/python-3.4.3-0/include/python3.4m -o \
    mongodb.so mongodb.c

echo "Running cython on datasets.pyx and then compiling with gcc..."
cython -a datasets.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I/opt/python/conda_default/pkgs/python-3.4.3-0/include/python3.4m -o \
    datasets.so datasets.c

echo "Complete.\n"
