#!/bin/bash -e
if [ $# != 2 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
KERNEL="$2"

trap "exit \$exit_code" INT TERM
trap "exit_code=\$?; kill 0" EXIT

CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 500 1 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 250 2 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 167 3 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 100 5 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 71 7 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 56 9 "${KERNEL}" 6000 &

CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 1 500 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 2 250 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 3 167 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 5 100 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 7 71 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 9 56 "${KERNEL}" 6000 &
wait

CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 45 11 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 38 13 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 33 15 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 29 17 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 26 19 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 24 21 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 22 23 "${KERNEL}" 6000 &

CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 11 45 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 13 38 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 15 33 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 17 29 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 19 26 "${KERNEL}" 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" 21 24 "${KERNEL}" 6000 &
wait
