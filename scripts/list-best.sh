#!/bin/bash

# Print the best candidate f-value from all hyperopt runs in the given folder.

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dir>..."
    exit 1
fi

for f in "$@"; do
    for out in "$f"/out*; do
        grep --text "Best candidate" "$out" | tail -n1
    done
done