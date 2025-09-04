#!/bin/bash
set -xeu

CODE_DIR=/lab-mlperf-inference/code
TEST06_DIR=/app/mlperf_inference/compliance/nvidia/TEST06
GPU_NAME=${GPU_NAME:-'16x_mi300x'}

if [ -f 'audit.config' ]; then
   rm audit.config
fi

## Offline
python $CODE_DIR/client_main.py --config-path $CODE_DIR/harness_llm/models/llama2-70b/ --config-name offline_$GPU_NAME test_mode=performance harness_config.output_log_dir=2_node_results/llama2-70b/Offline/performance/run_1
## Accuracy
python $CODE_DIR/client_main.py --config-path $CODE_DIR/harness_llm/models/llama2-70b/ --config-name offline_$GPU_NAME test_mode=accuracy    harness_config.output_log_dir=2_node_results/llama2-70b/Offline/accuracy
bash $CODE_DIR/scripts/check_llama2_accuracy_scores.sh 2_node_results/llama2-70b/Offline/accuracy/mlperf_log_accuracy.json
## Compliance TEST06
cp $TEST06_DIR/audit.config ./
python $CODE_DIR/client_main.py --config-path $CODE_DIR/harness_llm/models/llama2-70b/ --config-name offline_$GPU_NAME test_mode=performance harness_config.output_log_dir=2_node_results/llama2-70b/Offline/audit/compliance/TEST06
python3 $TEST06_DIR/run_verification.py -c 2_node_results/llama2-70b/Offline/audit/compliance/TEST06 -o 2_node_results/llama2-70b/Offline/audit/compliance -s Offline
rm audit.config