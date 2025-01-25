#!/bin/bash
set -e

# Download the datasets used in the experiments.

curl -L "https://drive.usercontent.google.com/download?id=1ieM39R5mpQz55nMub6LWSFmnSF4YJcg7&confirm=xxx" -o /var/tmp/datasets.tar.gz
tar -xvzf /var/tmp/datasets.tar.gz -C ./third_party/
rm /var/tmp/datasets.tar.gz