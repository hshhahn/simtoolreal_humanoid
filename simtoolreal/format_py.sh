#!/bin/bash

# Exit if no argument is provided
if [ -z "$1" ]; then
  echo "Usage: bash format_py.sh <path>"
  exit 1
fi

TARGET_PATH="$1"

# Run Ruff check and fix imports, then format
echo "Running Ruff on: $TARGET_PATH"
ruff check --extend-select I --fix "$TARGET_PATH"
ruff format "$TARGET_PATH"

echo "✅ Formatting complete for: $TARGET_PATH"

