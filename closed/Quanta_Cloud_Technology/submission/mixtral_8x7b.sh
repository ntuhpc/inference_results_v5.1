#!/bin/bash
set -xeu

CODE_DIR=/lab-mlperf-inference/code
SUBMISSION_DIR=/lab-mlperf-inference/submission
TEST06_DIR=/app/mlperf_inference/compliance/nvidia/TEST06
CPU_NAME=${CPU_NAME:-"EPYC_9554"} #lscpu | grep name
GPU_NAME=${GPU_NAME:-'mi300x'}
GPU_COUNT=${GPU_COUNT:-8}
COMPANY=${COMPANY:-'AMD'}
RESULTS=${RESULTS:-'results'}
ENABLE_POWER_SETUP=${ENABLE_POWER_SETUP:-'1'}
BACKEND=${BACKEND:-'vllm'}
OFFLINE=${OFFLINE:-'1'}
SERVER=${SERVER:-'1'}
INTERACTIVE=${INTERACTIVE:-'1'}
PERFORMANCE=${PERFORMANCE:-'1'}
ACCURACY=${ACCURACY:-'1'}
COMPLIANCE=${COMPLIANCE:-'1'}
PACKAGE=${PACKAGE:-'1'}

# Check if GPU_COUNT is either 1 or 8
if [[ "$GPU_COUNT" -ne 1 && "$GPU_COUNT" -ne 8 ]]; then
   echo "Error: GPU_COUNT must be set to 1 or 8. Current value: $GPU_COUNT"
   exit 1
fi

if [ -f 'audit.config' ]; then
   rm audit.config
fi

CPU_COUNT=$(lscpu | grep 'Socket(s)' | sed 's/[^0-9]*//g')
GPU_NAME_UPPER=$(echo "$GPU_NAME" | sed 's/.*/\U&/')
SYSTEM_NAME="${GPU_COUNT}x${GPU_NAME_UPPER}_${CPU_COUNT}x${CPU_NAME}"

# Get user.conf file path based on the GPU name and GPU count
USER_CONF_PATH="/lab-mlperf-inference/code/user_${GPU_NAME}"

if [[ "$GPU_COUNT" -eq 1 ]]; then
   USER_CONF_PATH+=_1gpu
fi

USER_CONF_PATH+=.conf

if [[ "$OFFLINE" == "1" ]]; then
if [[ "$ENABLE_POWER_SETUP" == "1" ]]; then
bash $CODE_DIR/scripts/power_settings.sh mixtral-8x7b $GPU_NAME offline
fi # ENABLE_POWER_SETUP
if [[ "$PERFORMANCE" == "1" ]]; then
python $CODE_DIR/main.py \
   --config-path $CODE_DIR/harness_llm/models/mixtral-8x7b/ \
   --config-name offline_$GPU_NAME \
   --backend ${BACKEND} \
   test_mode=performance \
   harness_config.device_count=${GPU_COUNT} \
   harness_config.user_conf_path=${USER_CONF_PATH} \
   harness_config.output_log_dir=$RESULTS/mixtral-8x7b/Offline/performance/run_1
fi # PERFORMANCE
if [[ "$ACCURACY" == "1" ]]; then
python $CODE_DIR/main.py \
   --config-path $CODE_DIR/harness_llm/models/mixtral-8x7b/ \
   --config-name offline_$GPU_NAME \
   --backend ${BACKEND} \
   test_mode=accuracy \
   harness_config.device_count=${GPU_COUNT} \
   harness_config.user_conf_path=${USER_CONF_PATH} \
   harness_config.output_log_dir=$RESULTS/mixtral-8x7b/Offline/accuracy

bash $CODE_DIR/scripts/setup_mixtral_accuracy_env.sh
bash $CODE_DIR/scripts/check_mixtral_accuracy_scores.sh \
   $RESULTS/mixtral-8x7b/Offline/accuracy/mlperf_log_accuracy.json
fi # ACCURACY
if [[ "$COMPLIANCE" == "1" ]]; then
## Compliance TEST06
cp $TEST06_DIR/audit.config ./
python $CODE_DIR/main.py \
   --config-path $CODE_DIR/harness_llm/models/mixtral-8x7b/ \
   --config-name offline_$GPU_NAME \
   --backend ${BACKEND} \
   test_mode=performance \
   harness_config.device_count=${GPU_COUNT} \
   harness_config.user_conf_path=${USER_CONF_PATH} \
   harness_config.output_log_dir=$RESULTS/mixtral-8x7b/Offline/audit/compliance/TEST06

python3 $TEST06_DIR/run_verification.py \
   -c $RESULTS/mixtral-8x7b/Offline/audit/compliance/TEST06 \
   -o $RESULTS/mixtral-8x7b/Offline/audit/compliance \
   -s Offline

rm audit.config
fi # COMPLIANCE
fi # OFFLINE

if [[ "$SERVER" == "1" ]]; then
if [[ "$ENABLE_POWER_SETUP" == "1" ]]; then
bash $CODE_DIR/scripts/power_settings.sh mixtral-8x7b $GPU_NAME server
fi # ENABLE_POWER_SETUP
if [[ "$PERFORMANCE" == "1" ]]; then
python $CODE_DIR/main.py \
   --config-path $CODE_DIR/harness_llm/models/mixtral-8x7b/ \
   --config-name server_$GPU_NAME \
   --backend ${BACKEND} \
   test_mode=performance \
   harness_config.device_count=${GPU_COUNT} \
   harness_config.user_conf_path=${USER_CONF_PATH} \
   harness_config.output_log_dir=$RESULTS/mixtral-8x7b/Server/performance/run_1
fi # PERFORMANCE
if [[ "$ACCURACY" == "1" ]]; then
python $CODE_DIR/main.py \
   --config-path $CODE_DIR/harness_llm/models/mixtral-8x7b/ \
   --config-name server_$GPU_NAME \
   --backend ${BACKEND} \
   test_mode=accuracy \
   harness_config.device_count=${GPU_COUNT} \
   harness_config.user_conf_path=${USER_CONF_PATH} \
   harness_config.output_log_dir=$RESULTS/mixtral-8x7b/Server/accuracy

bash $CODE_DIR/scripts/setup_mixtral_accuracy_env.sh
bash $CODE_DIR/scripts/check_mixtral_accuracy_scores.sh \
   $RESULTS/mixtral-8x7b/Server/accuracy/mlperf_log_accuracy.json
fi # ACCURACY
if [[ "$COMPLIANCE" == "1" ]]; then
## Compliance TEST06
cp $TEST06_DIR/audit.config ./
python $CODE_DIR/main.py \
   --config-path $CODE_DIR/harness_llm/models/mixtral-8x7b/ \
   --config-name server_$GPU_NAME \
   --backend ${BACKEND} \
   test_mode=performance \
   harness_config.device_count=${GPU_COUNT} \
   harness_config.user_conf_path=${USER_CONF_PATH} \
   harness_config.output_log_dir=$RESULTS/mixtral-8x7b/Server/audit/compliance/TEST06

python3 $TEST06_DIR/run_verification.py \
   -c $RESULTS/mixtral-8x7b/Server/audit/compliance/TEST06 \
   -o $RESULTS/mixtral-8x7b/Server/audit/compliance \
   -s Server

rm audit.config
fi # COMPLIANCE
fi # SERVER


if [ -d "$CODE_DIR/moe_accuracy_venv" ]; then rm -r $CODE_DIR/moe_accuracy_venv; fi

if [[ "$PACKAGE" == "1" ]]; then
. $SUBMISSION_DIR/package_submission.sh
fi # PACKAGE
