#!/bin/zsh

# Usage: ./cythonize.sh

UTIL_DIR=$(dirname $(readlink -f $0))
PROJECT_DIR=$(dirname ${UTIL_DIR})
SRC_DIR=${PROJECT_DIR}/src

ROOTENV=$(conda info | grep "root environment :" | awk '{print $4}')
PYTHON_HEADER_DIR=${ROOTENV}/pkgs/python-3.4.3-0/include/python3.4m

echo "Running cython on feature_extraction.pyx and then compiling with" \
    "gcc..."
echo "cython -a ${SRC_DIR}/feature_extraction.pyx"
cython -a ${SRC_DIR}/feature_extraction.pyx
echo "gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I${PYTHON_HEADER_DIR} -o ${SRC_DIR}/feature_extraction.so ${SRC_DIR}/feature_extraction.c"
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I${PYTHON_HEADER_DIR} -o ${SRC_DIR}/feature_extraction.so \
    ${SRC_DIR}/feature_extraction.c

echo "Running cython on mongodb.pyx and then compiling with gcc..."
echo "cython -a ${UTIL_DIR}/mongodb.pyx"
cython -a ${UTIL_DIR}/mongodb.pyx
echo "gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I${PYTHON_HEADER_DIR} -o ${UTIL_DIR}/mongodb.so ${UTIL_DIR}/mongodb.c"
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I${PYTHON_HEADER_DIR} -o ${UTIL_DIR}/mongodb.so ${UTIL_DIR}/mongodb.c

echo "Running cython on datasets.pyx and then compiling with gcc..."
echo "cython -a ${UTIL_DIR}/datasets.pyx"
cython -a ${UTIL_DIR}/datasets.pyx
echo "gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I${PYTHON_HEADER_DIR} -o ${UTIL_DIR}/datasets.so ${UTIL_DIR}/datasets.c"
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I${PYTHON_HEADER_DIR} -o ${UTIL_DIR}/datasets.so \
    ${UTIL_DIR}/datasets.c

echo "\nComplete.\n"
