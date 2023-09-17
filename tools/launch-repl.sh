#!/bin/bash

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$TOOLS_DIR")"

LESSON_1_DIR="$ROOT_DIR/lesson-1--build-micrograd"

export PYTHONPATH="$LESSON_1_DIR:$PYTHONPATH"

source "$ROOT_DIR/venv/bin/activate"

ptpython --vi
