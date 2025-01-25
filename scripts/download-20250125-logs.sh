#!/bin/bash
set -e

# Download the logs and snapshots published with the Locally
# Connected Echo State Networks for Time Series Forecasting paper.

curl -L "https://drive.usercontent.google.com/download?id=1aZip6t1ZblANVqbYk-RNCAOBwzN-iYpO&confirm=xxx" -o /var/tmp/20250125-logs-published.tar.xz
tar -xvJf /var/tmp/20250125-logs-published.tar.xz -C ./
rm /var/tmp/20250125-logs-published.tar.xz