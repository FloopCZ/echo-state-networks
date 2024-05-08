#!/bin/bash
curl -L "https://drive.usercontent.google.com/download?id=1ieM39R5mpQz55nMub6LWSFmnSF4YJcg7&confirm=xxx" -o /tmp/datasets.tar.gz
tar -xvzf /tmp/datasets.tar.gz -C ./third_party/
rm /tmp/datasets.tar.gz