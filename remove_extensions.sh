#!/bin/zsh
# Removes the build directory and all files associated with extensions (except
# the extension modules themselves!).

# Usage: ./remove_extensions.sh

rm -fR build src/*c src/*so util/*so util/*c &
