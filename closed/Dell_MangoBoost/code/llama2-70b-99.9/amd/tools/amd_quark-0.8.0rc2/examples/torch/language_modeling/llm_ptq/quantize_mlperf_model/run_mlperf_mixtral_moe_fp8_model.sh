#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

#!/bin/bash

export HF_HOME=/quark/hf_cache


DATASET="/model/mlperf_data/mixtral_8x7b%2F2024.06.06_mixtral_15k_calibration_v4.pkl"
MODEL="/model/mistralai/Mixtral-8x7B-Instruct-v0.1-MLCommons"


OUTPUT_DIR="/model/mistralai/Mixtral-8x7B-Instruct-v0.1-MLCommons-FP8-MLPerf-V1-retest"   # [INFO] Perplexity: 4.184244155883789

python3 quantize_quark.py --model_dir "${MODEL}" \
                          --output_dir "${OUTPUT_DIR}" \
                          --dataset "${DATASET}" \
                          --data_type float16 \
                          --multi_gpu \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --num_calib_data 1024 \
                          --seq_len 1024 \
                          --min_kv_scale 1.0 \
                          --model_export hf_format \
                          --custom_mode fp8 \
                          --exclude_layers "lm_head" "*.gate" "*q_proj" "*k_proj" "*v_proj" "*o_proj"


OUTPUT_DIR="/model/mistralai/Mixtral-8x7B-Instruct-v0.1-MLCommons-FP8-MLPerf-V2-retest"    # [INFO] Perplexity: 4.20029354095459

python3 quantize_quark.py --model_dir "${MODEL}" \
                          --output_dir "${OUTPUT_DIR}" \
                          --dataset "${DATASET}" \
                          --data_type float16 \
                          --multi_gpu \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --num_calib_data 1024 \
                          --seq_len 1024 \
                          --min_kv_scale 1.0 \
                          --model_export hf_format \
                          --custom_mode fp8 \
                          --quant_algo autosmoothquant \
                          --exclude_layers "lm_head" "*.gate" "*o_proj"

OUTPUT_DIR="/model/mistralai/Mixtral-8x7B-Instruct-v0.1-MLCommons-FP8-MLPerf-V3-retest"   # [INFO] Perplexity: 4.202735424041748

python3 quantize_quark.py --model_dir "${MODEL}" \
                          --output_dir "${OUTPUT_DIR}" \
                          --dataset "${DATASET}" \
                          --data_type float16 \
                          --multi_gpu \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --num_calib_data 1024 \
                          --seq_len 1024 \
                          --min_kv_scale 1.0 \
                          --model_export hf_format \
                          --custom_mode fp8 \
                          --quant_algo autosmoothquant \
                          --exclude_layers "lm_head" "*.gate"
