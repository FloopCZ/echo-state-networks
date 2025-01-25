#!/bin/bash

# This script runs a given command on a free GPU. If no GPU is available, it waits until one becomes available.
# The script assumes that the command uses the CUDA_VISIBLE_DEVICES environment variable to select the GPU.

# The script creates a lock file for each GPU with the PID of the process using the GPU. The lock file is removed
# when the process finishes. If the process dies unexpectedly, the lock file is removed and the GPU is considered free.

if [ $# -lt 1 ]; then
    echo "Usage: $0 <command> [<arg1> <arg2> ...]"
    exit 1
fi

LOCK_DIR="/tmp/gpu_lock"
MAX_MEM=100

mkdir -p $LOCK_DIR
chmod 777 $LOCK_DIR

# Function to get a list of free GPUs (i.e., GPUs using less than 100MB of memory).
get_free_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | while read -r line; do
        index=$(echo $line | cut -d ',' -f 1 | xargs)
        memory=$(echo $line | cut -d ',' -f 2 | xargs)
        if (( memory < $MAX_MEM )); then
            if [ ! -f $LOCK_DIR/gpu_$index.lock ]; then
                echo $index
            else
                pid=$(cat $LOCK_DIR/gpu_$index.lock)
                if ! kill -0 $pid 2>/dev/null; then
                    rm $LOCK_DIR/gpu_$index.lock
                    echo $index
                fi
            fi
        fi
    done
}

# Function to run the task on a given GPU.
run_task_on_gpu() {
    local gpu_id=$1
    shift
    echo "Using GPU $gpu_id to run:"
    echo "$@"
    CUDA_VISIBLE_DEVICES=$gpu_id "$@"
    echo "GPU $gpu_id finished."
}

# Function to acquire a lock for a specific GPU
acquire_lock() {
    local gpu_id=$1
    if ( set -o noclobber; echo "$$" > $LOCK_DIR/gpu_$gpu_id.lock ) 2> /dev/null; then
        chmod 666 $LOCK_DIR/gpu_$gpu_id.lock
        return 0
    else
        return 1
    fi
}

# Function to release a lock for a specific GPU
release_lock() {
    local gpu_id=$1
    rm -f $LOCK_DIR/gpu_$gpu_id.lock
}

while true; do
    free_gpus=($(get_free_gpus))
    if [ ${#free_gpus[@]} -gt 0 ]; then
        gpu_id=${free_gpus[0]}
        if acquire_lock $gpu_id; then
            run_task_on_gpu $gpu_id "$@"
            rv=$?
            release_lock $gpu_id
            exit $rv
        else
            echo "Failed to acquire lock for GPU $gpu_id. Trying next available GPU."
        fi
    fi
    echo "No available gpu."
    sleep 30
done

