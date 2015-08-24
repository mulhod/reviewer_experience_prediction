#!/bin/zsh
# Removes the build directory and all files associated with extensions (except
# the extension modules themselves!).

# Usage: ./remove_extensions.sh

THIS_DIR=$(dirname $(readlink -f $0))
if [[ -e ${THIS_DIR}/build ]]; then
    
    rm -fR ${THIS_DIR}/build
    
fi
if [[ $(ls ${THIS_DIR}/src | grep -P "\.c$" | wc -l) -gt 0 ]]; then
    
    rm -f ${THIS_DIR}/src/*.c
    
fi
if [[ $(ls ${THIS_DIR}/src | grep -P "\.so$" | wc -l) -gt 0 ]]; then

    rm -f ${THIS_DIR}/src/*.so

fi
if [[ $(ls ${THIS_DIR}/util | grep -P "\.c$" | wc -l) -gt 0 ]]; then

    rm -f ${THIS_DIR}/util/*.c

fi
if [[ $(ls ${THIS_DIR}/util | grep -P "\.so$" | wc -l) -gt 0 ]]; then

    rm -f ${THIS_DIR}/util/*.so

fi
if [[ $(ls ${THIS_DIR} | grep -P "\.c$" | wc -l) -gt 0 ]]; then

    rm -f ${THIS_DIR}/*.c

fi
if [[ $(ls ${THIS_DIR} | grep -P "\.so$" | wc -l) -gt 0 ]]; then

    rm -f ${THIS_DIR}/*.so

fi
