#!/bin/bash
set -e

max_mem=0

terminate_script() {
    echo "Max used GPU memory: ${max_mem} MiB"
    exit 0
}
trap terminate_script SIGINT

while true; do
    curr_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
    if (( curr_mem > max_mem )); then
        max_mem=$curr_mem
    fi
    sleep 1
done