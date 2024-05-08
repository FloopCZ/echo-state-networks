#!/bin/bash

logdir=${1:-"./log"}

for d in ${logdir}/optimize-*; do
    echo "$d"
    ./scripts/list-best.sh "$d"
done