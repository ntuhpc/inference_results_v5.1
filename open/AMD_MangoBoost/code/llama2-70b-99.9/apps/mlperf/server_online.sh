#!/bin/bash
set -xeu

python3 server.py \
    -tp 1 \
    -dp 8 \
    -max_num_seqs 768 \
    --gpu_batch_size 48 \
    --batcher_threshold 0.2 \
    --load_balancing_mode batching \
    --test_mode Server
