#!/bin/bash

log_dir=${1:-"./log"}

for d in ${log_dir}/optimize-*; do
    echo "$d"
    ./scripts/list-best.sh "$d"
done