#!/bin/bash
set -e

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 0

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 15 0.5
./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 15 1.0

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 30 0.5
./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 30 1.0

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 60 0.5
./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 60 1.0

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 120 0.5
./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 120 1.0

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 240 0.5
./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 240 1.0

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 480 0.5
./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MEMLEN-MEMPROB-loop-cluster.sh lcnn 40 50 336 480 1.0