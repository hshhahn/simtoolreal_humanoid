#!/bin/bash

find . -name "*.urdf" -type f -exec xmllint --format "{}" --output "{}" \;