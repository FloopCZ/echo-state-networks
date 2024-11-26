#!/bin/bash

# List the best candidates from all hyperopt runs in the log/ folder.

log_dir=${1:-"./log"}

for d in ${log_dir}/optimize-*; do
    echo "$d"
    ./scripts/list-best.sh "$d"
done