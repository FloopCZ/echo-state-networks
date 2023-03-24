#!/bin/bash -e
if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 500 1 6000 &
CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 167 3 6000 &
CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 100 5 6000 &
CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 71 7 6000 &
CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 56 9 6000 &
CUDA_VISIBLE_DEVICES="1" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 1 500 6000 &
CUDA_VISIBLE_DEVICES="1" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 3 167 6000 &
CUDA_VISIBLE_DEVICES="1" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 5 100 6000 &
CUDA_VISIBLE_DEVICES="1" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 7 71 6000 &
CUDA_VISIBLE_DEVICES="1" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 9 56 6000 &
wait

CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 45 11 6000 &
CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 38 13 6000 &
CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 33 15 6000 &
CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 29 17 6000 &
CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 26 19 6000 &
CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 24 21 6000 &
CUDA_VISIBLE_DEVICES="0" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 22 23 6000 &
CUDA_VISIBLE_DEVICES="1" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 11 45 6000 &
CUDA_VISIBLE_DEVICES="1" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 13 38 6000 &
CUDA_VISIBLE_DEVICES="1" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 15 33 6000 &
CUDA_VISIBLE_DEVICES="1" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 17 29 6000 &
CUDA_VISIBLE_DEVICES="1" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 19 26 6000 &
CUDA_VISIBLE_DEVICES="1" ./build/compare-TOPO-HEIGHT-WIDTH-TRAIN-kernels-narma10-v1.sh "${TOPO}" 21 24 6000 &
wait
