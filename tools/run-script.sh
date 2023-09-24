#!/bin/bash

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$TOOLS_DIR")"

LECTURE_1_DIR="$ROOT_DIR/lecture-1--build-micrograd"
export PYTHONPATH="$LECTURE_1_DIR:$PYTHONPATH"

source "$ROOT_DIR/venv/bin/activate"

python ${@:1}
