#!/bin/bash
set -e

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MUMEMLEN-loop-cluster.sh lcnn 40 50 336 336

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MUMEMLEN-loop-cluster.sh lcnn 40 50 336 192

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MUMEMLEN-loop-cluster.sh lcnn 40 50 336 96

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MUMEMLEN-loop-cluster.sh lcnn 40 50 192 192

./experiments/optimize-TOPO-HEIGHT-WIDTH-ettm1-AHEAD-MUMEMLEN-loop-cluster.sh lcnn 40 50 192 96