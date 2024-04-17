#!/bin/bash
set -e

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 192 240 0.5
./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 192 240 1.0

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 192 120 0.5
./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 192 120 1.0

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 192 60 0.5
./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 192 60 1.0

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 192 30 0.5
./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 192 30 1.0

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 192 15 0.5
./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 192 15 1.0

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 192 0 0.0