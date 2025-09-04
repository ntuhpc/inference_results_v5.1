#!/bin/bash
set -xeu

CODE_DIR=/lab-mlperf-inference/code
SUBMISSION_DIR=/lab-mlperf-inference/submission
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
SYSTEM_NAME=${SYSTEM_NAME:-"8xMI300X_2xEPYC_95"} #CPU name: lscpu | grep name; count: lscpu | grep 'node(s)'
GPU_NAME=${GPU_NAME:-'8x_mi300x'}
COMPANY=${COMPANY:-'MangoBoost'}

if [ -f 'audit.config' ]; then
   rm audit.config
fi

# Offline
## Perf
python $CODE_DIR/main.py --config-path $CODE_DIR/harness_llm/models/llama2-70b/ --config-name offline_$GPU_NAME test_mode=performance harness_config.output_log_dir=results/llama2-70b/Offline/performance/run_1
## Accuracy
python $CODE_DIR/main.py --config-path $CODE_DIR/harness_llm/models/llama2-70b/ --config-name offline_$GPU_NAME test_mode=accuracy    harness_config.output_log_dir=results/llama2-70b/Offline/accuracy
bash $CODE_DIR/scripts/check_llama2_accuracy_scores.sh results/llama2-70b/Offline/accuracy/mlperf_log_accuracy.json
## Compliance TEST06
cp $TEST06_DIR/audit.config ./
python $CODE_DIR/main.py --config-path $CODE_DIR/harness_llm/models/llama2-70b/ --config-name offline_$GPU_NAME test_mode=performance harness_config.output_log_dir=results/llama2-70b/Offline/audit/compliance/TEST06
python3 $TEST06_DIR/run_verification.py -c results/llama2-70b/Offline/audit/compliance/TEST06 -o results/llama2-70b/Offline/audit/compliance -s Offline
rm audit.config

# Server
# Perf
python $CODE_DIR/main.py --config-path $CODE_DIR/harness_llm/models/llama2-70b/ --config-name server_$GPU_NAME test_mode=performance harness_config.output_log_dir=results/llama2-70b/Server/performance/run_1
# ## Accuracy
python $CODE_DIR/main.py --config-path $CODE_DIR/harness_llm/models/llama2-70b/ --config-name server_$GPU_NAME test_mode=accuracy    harness_config.output_log_dir=results/llama2-70b/Server/accuracy
bash $CODE_DIR/scripts/check_llama2_accuracy_scores.sh results/llama2-70b/Server/accuracy/mlperf_log_accuracy.json
## Compliance TEST06
cp $TEST06_DIR/audit.config ./
python $CODE_DIR/main.py --config-path $CODE_DIR/harness_llm/models/llama2-70b/ --config-name server_$GPU_NAME test_mode=performance harness_config.output_log_dir=results/llama2-70b/Server/audit/compliance/TEST06
python3 $TEST06_DIR/run_verification.py -c results/llama2-70b/Server/audit/compliance/TEST06 -o results/llama2-70b/Server/audit/compliance -s Server
rm audit.config

# Package submission
python $SUBMISSION_DIR/package_submission.py --base-package-dir inference_results_5.0 --input-dir results/llama2-70b --scenarios Server Offline --system-name $SYSTEM_NAME --benchmark llama2-70b-99   --user-conf /lab-mlperf-inference/code/user_$GPU_NAME.conf --company ${COMPANY}
python $SUBMISSION_DIR/package_submission.py --base-package-dir inference_results_5.0 --input-dir results/llama2-70b --scenarios Server Offline --system-name $SYSTEM_NAME --benchmark llama2-70b-99.9 --user-conf /lab-mlperf-inference/code/user_$GPU_NAME.conf --company ${COMPANY}
