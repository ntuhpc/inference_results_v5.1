#!/bin/bash
set -xeu

CODE_DIR=/lab-mlperf-inference/code
GPU_NAME=${GPU_NAME:-'16x_mi300x'}

if [ -f 'audit.config' ]; then
   rm audit.config
fi

# Offline
python $CODE_DIR/server_main.py --config-path $CODE_DIR/harness_llm/models/llama2-70b/ --config-name offline_$GPU_NAME test_mode=performance harness_config.output_log_dir=2_node_results/llama2-70b/Offline/performance/run_1
