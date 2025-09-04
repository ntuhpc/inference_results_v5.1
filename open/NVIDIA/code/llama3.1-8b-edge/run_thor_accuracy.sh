#!/bin/bash

# Get scenario from command line argument, default to SingleStream if not provided
SCENARIO=${1:-SingleStream}


export ATTENTION_PLUGIN_PATH=/tmp/build/libAttentionPlugin.so
export INT4_GEMM_PLUGIN_PATH=/tmp/build/libInt4GemmPlugin.so
ENGINE_PATH=/home/engines/new_docker/8B_W4A16.engine
OUTPUT_DIR=MLPerf_submission_W4A16_accuracy_5000_dataset


python main.py --scenario $SCENARIO --model-path=/home/engines/Meta-Llama-3.1-8B-Instruct \
 --dataset-path dataset/5000/sample_cnn_eval_5000.json \
 --batch-size=1024 \
 --user-conf=user.conf \
 --output-log-dir=$OUTPUT_DIR \
 --enable-log-trace \
 --total-sample-count 5000 \
 --token-output-file=$OUTPUT_DIR/tokens_output.csv \
 --engine-dir=$ENGINE_PATH \
 --async-mode \
 --accuracy
