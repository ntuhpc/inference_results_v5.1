#!/bin/bash

cd amd_quark-0.9/examples/torch/language_modeling/llm_ptq

ARCH=$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')
DATASET="/data/llama3.1-405b/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl"
MODEL="/model/llama3.1-405b/orig"
OUTPUT_DIR="/model/llama3.1-405b/fp8_quantized"

if [[ "$ARCH" == "gfx950" ]]; then
    OUTPUT_DIR="/model/llama3.1-405b/fp4_quantized"
    python3 quantize_quark.py --model_dir "${MODEL}" \
                          --output_dir "${OUTPUT_DIR}" \
                          --dataset "${DATASET}" \
                          --model_attn_implementation "sdpa" \
                          --seq_len 2048 \
                          --data_type auto \
                          --quant_algo autosmoothquant \
                          --quant_scheme w_mxfp4_a_mxfp4 \
                          --group_size 32 \
                          --kv_cache_dtype fp8 \
                          --min_kv_scale 1.0 \
                          --exclude_layers "lm_head" \
                          --model_export hf_format \
                          --multi_gpu
else
    python3 quantize_quark.py --model_dir "${MODEL}" \
                            --output_dir "${OUTPUT_DIR}" \
                            --dataset "${DATASET}" \
                            --multi_gpu \
                            --data_type auto \
                            --model_attn_implementation "sdpa" \
                            --quant_algo autosmoothquant \
                            --quant_scheme w_fp8_a_fp8 \
                            --kv_cache_dtype fp8 \
                            --min_kv_scale 1.0 \
                            --num_calib_data 512 \
                            --seq_len 8192 \
                            --model_export hf_format \
                            --custom_mode fp8 \
                            --exclude_layers "lm_head"
fi