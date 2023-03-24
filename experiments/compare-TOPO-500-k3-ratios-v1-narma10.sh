#!/bin/bash -e
if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

trap "exit \$exit_code" INT TERM
trap "exit_code=\$?; kill 0" EXIT

CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 500 1 3 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 167 3 3 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 100 5 3 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 71 7 3 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 56 9 3 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 45 11 3 6000 &

CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 1 500 3 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 3 167 3 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 5 100 3 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 7 71 3 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 9 56 3 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 11 45 3 6000 &
wait

CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 38 13 3 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 33 15 3 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 29 17 3 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 26 19 3 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 24 21 3 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 22 23 3 6000 &

CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 13 38 3 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 15 33 3 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 17 29 3 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 19 26 3 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 21 24 3 6000 &
wait
