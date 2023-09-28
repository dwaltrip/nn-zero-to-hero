#!/bin/bash

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$TOOLS_DIR")"

cd $ROOT_DIR
source "$ROOT_DIR/venv/bin/activate"

python -m jupyter notebook
