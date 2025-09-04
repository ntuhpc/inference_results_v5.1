#!/bin/bash

cd llm_ptq

DATASET="/data/processed-openorca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl"
MODEL="/model/llama2-70b-chat-hf/orig"
OUTPUT_DIR="/model/llama2-70b-chat-hf/fp8_quantized"

python3 quantize_quark.py --model_dir "${MODEL}" \
                          --output_dir "${OUTPUT_DIR}" \
                          --dataset "${DATASET}" \
                          --data_type float16 \
                          --multi_gpu \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --num_calib_data 1000 \
                          --seq_len 1024 \
                          --model_export hf_format \
                          --custom_mode fp8 \
                          --exclude_layers "lm_head"
