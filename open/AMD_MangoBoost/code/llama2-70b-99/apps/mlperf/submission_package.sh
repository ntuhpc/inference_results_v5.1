#!/bin/bash
set -xeu

SUBMISSION_DIR=/workspace/apps/mlperf/submission
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
SYSTEM_NAME=${SYSTEM_NAME:-"32xMI300X_2xEPYC_9534"} #CPU name: lscpu | grep name; count: lscpu | grep 'node(s)'
GPU_NAME=${GPU_NAME:-'mi300x'}
COMPANY=${COMPANY:-'MangoBoost'}


## Package the results
python $SUBMISSION_DIR/package_submission.py --base-package-dir inference_results_5.0 --input-dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME --scenarios Server Offline --system-name $SYSTEM_NAME --benchmark llama2-70b-99   --user-conf conf/user_8x_mi300x.conf --company ${COMPANY} --system-description systems/32x_mi300_system.json
python $SUBMISSION_DIR/package_submission.py --base-package-dir inference_results_5.0 --input-dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME --scenarios Server Offline --system-name $SYSTEM_NAME --benchmark llama2-70b-99.9 --user-conf conf/user_8x_mi300x.conf --company ${COMPANY} --system-description systems/32x_mi300_system.json
