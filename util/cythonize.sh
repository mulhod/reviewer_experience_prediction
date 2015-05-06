#!/bin/zsh

# To be used on lemur.montclair.edu only

source /home/mulhollandm2/.zshrc
source activate reviews

REVIEWS=/home/mulhollandm2/reviews_project/reviewer_experience_prediction
cd ${REVIEWS}

echo "Running cython on feature_extraction.pyx and then compiling with" \
    "gcc...\n"
cd src
cython -a feature_extraction.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I/home/mulhollandm2/conda/envs/reviews/include/python3.4m -o \
    feature_extraction.so feature_extraction.c

echo "Running cython on mongodb.pyx and then compiling with gcc...\n"
cd ../util
cython -a mongodb.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I/home/mulhollandm2/conda/envs/reviews/include/python3.4m -o \
    mongodb.so mongodb.c

echo "Running cython on datasets.pyx and then compiling with gcc...\n"
cython -a datasets.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I/home/mulhollandm2/conda/envs/reviews/include/python3.4m -o \
    datasets.so datasets.c

echo "Complete.\n"