#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

#!/bin/bash

export HF_HOME=/quark/hf_cache

DATASET="/model/mlperf_data/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl"
MODEL="/model/meta-llama/Meta-Llama-3.1-405B-Instruct"

# Meta-Llama-3.1-405B-Instruct_fp8_quantized without fp8 attention

OUTPUT_DIR="/model/meta-llama/Meta-Llama-3.1-405B-Instruct_fp8_quantized_seqlen_10000_autosq_retest"


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
