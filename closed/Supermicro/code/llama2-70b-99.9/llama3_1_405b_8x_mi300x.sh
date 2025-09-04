#!/bin/bash
set -xeu

SUBMISSION_DIR=/workspace/apps/mlperf/submission
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
SYSTEM_NAME=${SYSTEM_NAME:-"8xMI300X_2xEPYC_9534"} #CPU name: lscpu | grep name; count: lscpu | grep 'node(s)'

if [ -f 'audit.config' ]; then
   rm audit.config
fi

# Offline
## Performance
python3 mlperf.py \
    --model_name llama3_1-405b \
    --test_mode Offline \
    -tp 8 \
    -dp 1 \
    --result_dir $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Offline/performance/run_1 \
    --user_conf conf/user_llama3_1-405b_8x_mi300x.conf \
    --max_model_len 131072
## Accuracy
python3 mlperf.py \
    --model_name llama3_1-405b \
    --test_mode Offline \
    -tp 8 \
    -dp 1 \
    --result_dir $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Offline/accuracy \
    --user_conf conf/user_llama3_1-405b_8x_mi300x.conf \
    --max_model_len 131072 \
    --accuracy_test 
bash tools/check_llama3_1_405B_accuracy_scores.sh $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Offline/accuracy/mlperf_log_accuracy.json
## Compliance TEST06
cp $TEST06_DIR/audit.config ./
python3 mlperf.py \
    --model_name llama3_1-405b \
    --test_mode Offline \
    -tp 8 \
    -dp 1 \
    --max_model_len 131072 \
    --result_dir $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Offline/audit/compliance/TEST06
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Offline/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Offline/audit/compliance -s Offline
rm audit.config

# Server
## Performance
python3 mlperf.py \
   --model_name llama3_1-405b \
   --test_mode Server \
   -tp 8 \
   -dp 1 \
   --drain_per_worker \
   --gpu_batch_size 48 \
   --batcher_threshold 0.2 \
   --load_balancing_mode batching \
   --result_dir $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Server/performance/run_1 \
   --max_model_len 131072 \
   --user_conf conf/user_llama3_1-405b_8x_mi300x.conf

## Accuracy
python3 mlperf.py \
   --model_name llama3_1-405b \
   --test_mode Server \
   -tp 8 \
   -dp 1 \
   --drain_per_worker \
   --gpu_batch_size 48 \
   --batcher_threshold 0.2 \
   --load_balancing_mode batching \
   --result_dir $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Server/accuracy \
   --user_conf conf/user_llama3_1-405b_8x_mi300x.conf \
   --max_model_len 131072 \
   --accuracy_test
bash tools/check_llama3_1_405B_accuracy_scores.sh $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Server/accuracy/mlperf_log_accuracy.json
## Compliance TEST06
cp $TEST06_DIR/audit.config ./
python3 mlperf.py \
   --model_name llama3_1-405b \
   --test_mode Server \
   -tp 8 \
   -dp 1 \
   --drain_per_worker \
   --gpu_batch_size 48 \
   --batcher_threshold 0.2 \
   --load_balancing_mode batching \
   --max_model_len 131072 \
   --result_dir $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Server/audit/compliance/TEST06
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Server/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama3_1-405b/$SYSTEM_NAME/Server/audit/compliance -s Server
rm audit.config
