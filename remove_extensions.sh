#!/bin/zsh
# Removes the build directory and all files associated with extensions (except the extension modules themselves!).
# Also removes *pyc and __pycache__ files/directories.

# Usage: ./remove_extensions.sh

rm -fR build src/*c src/*so src/*pyc src/__pycache__ util/*so util/*c util/*pyc util/__pycache__ &