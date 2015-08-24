#!/bin/zsh
# Removes the build directory and all files associated with extensions (except
# the extension modules themselves!).

# Usage: ./remove_extensions.sh

THIS_DIR=$(readlink -f $0)
rm -fR ${THIS_DIR}/build ${THIS_DIR}/src/*.c ${THIS_DIR}/src/*.so \
    ${THIS_DIR}/util/*so util/*.c ${THIS_DIR}/*.so &
