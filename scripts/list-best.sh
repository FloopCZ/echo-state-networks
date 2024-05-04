#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dir>..."
    exit 1
fi

for f in "$@"; do
    grep --text "Best candidate" "$f"/out* | tail -n1
done