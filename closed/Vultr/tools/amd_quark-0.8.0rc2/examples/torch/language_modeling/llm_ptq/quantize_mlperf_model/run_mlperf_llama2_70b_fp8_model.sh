#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

#!/bin/bash

export HF_HOME=/quark/hf_cache


DATASET="/model/mlperf_data/open_orca_gpt4_tokenized_llama.calibration_1000.pkl"
MODEL="/model/meta-llama/Llama-2-70b-chat-hf"

# Llama-2-70b-chat-hf_mlperf_fp8_quantized_model without fp8 attention

OUTPUT_DIR="/model/meta-llama/Llama-2-70b-chat-hf_FP8_MLPerf_V2_retest"   # [INFO] Perplexity: 4.6688079833984375

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
