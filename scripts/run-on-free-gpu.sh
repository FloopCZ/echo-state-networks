#!/bin/bash

LOCK_DIR="/tmp/gpu_lock"
mkdir -p $LOCK_DIR

# Function to get a list of free GPUs (i.e., GPUs using less than 100MB of memory).
get_free_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | while read -r line; do
        index=$(echo $line | cut -d ',' -f 1 | xargs)
        memory=$(echo $line | cut -d ',' -f 2 | xargs)
        if (( memory < 100 )); then
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

