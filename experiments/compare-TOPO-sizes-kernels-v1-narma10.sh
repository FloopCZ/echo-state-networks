#!/bin/bash -e
if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

trap "exit \$exit_code" INT TERM
trap "exit_code=\$?; kill 0" EXIT

CUDA_VISIBLE_DEVICES="0" ./experiments/compare-TOPO-HEIGHT-WIDTH-kernels-TRAIN-v1-narma10.sh lcnn-od 20 25 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/compare-TOPO-HEIGHT-WIDTH-kernels-TRAIN-v1-narma10.sh lcnn-od 28 36 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/compare-TOPO-HEIGHT-WIDTH-kernels-TRAIN-v1-narma10.sh lcnn-od 40 50 6000 &
CUDA_VISIBLE_DEVICES="1" ./experiments/compare-TOPO-HEIGHT-WIDTH-kernels-TRAIN-v1-narma10.sh lcnn-od 57 70 9000 &
CUDA_VISIBLE_DEVICES="0" ./experiments/compare-TOPO-HEIGHT-WIDTH-kernels-TRAIN-v1-narma10.sh lcnn-od 80 100 12000 &
wait
