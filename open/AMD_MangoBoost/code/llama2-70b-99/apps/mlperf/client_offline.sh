#!/bin/bash
set -xeu

SUBMISSION_DIR=/workspace/apps/mlperf/submission
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
SYSTEM_NAME=${SYSTEM_NAME:-"32xMI300X_2xEPYC_9534"} #CPU name: lscpu | grep name; count: lscpu | grep 'node(s)'
GPU_NAME=${GPU_NAME:-'mi300x'}
COMPANY=${COMPANY:-'MangoBoost'}

if [ -f 'audit.config' ]; then
   rm audit.config
fi

## Perf
python3 client.py \
    --test_mode Offline \
    --model_name llama2-70b \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/performance/run_1 \
    --user_conf conf/user_32x_mi300x.conf \
    --batched_queries 128 \
    --sut_server_addr "http://10.4.16.1:8000,http://10.4.16.2:8000,http://10.4.16.3:8000,http://10.4.16.4:8000"

## Accuracy
python3 client.py \
    --test_mode Offline \
    --model_name llama2-70b \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy  \
    --user_conf conf/user_32x_mi300.conf \
    --accuracy_test \
    --batched_queries 128 \
    --sut_server_addr="http://10.4.16.1:8000,http://10.4.16.2:8000,http://10.4.16.3:8000,http://10.4.16.4:8000"
bash tools/check_llama2_accuracy_scores.sh $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy/mlperf_log_accuracy.json

# Audit
cp $TEST06_DIR/audit.config ./
python3 client.py \
    --test_mode Offline \
    --model_name llama2-70b \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06 \
    --batched_queries 128 \
    --sut_server_addr="http://10.4.16.1:8000,http://10.4.16.2:8000,http://10.4.16.3:8000,http://10.4.16.4:8000"
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance -s Offline
rm audit.config