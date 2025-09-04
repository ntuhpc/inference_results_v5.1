#!/bin/bash

DATA_DIR=${DATA_DIR:-/data/llama3.1-405b}

if [ -e /lab-mlperf-inference/setup/download_llama3_405b.sh ]
then
    echo "Inside container, start downloading..."
    mkdir -p ${DATA_DIR}

    if [ -e "${DATA_DIR}/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl" ]
    then
        echo "Benchmark dataset for Llama3.1-405B is already exist"
    else
        rclone copy mlc-inference:mlcommons-inference-wg-public/llama3.1_405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl "${DATA_DIR}" -P
    fi

    if [ -e "${DATA_DIR}/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl" ]
    then
        echo "Calibration dataset for Llama3.1-405B is already exist"
    else
        rclone copy mlc-inference:mlcommons-inference-wg-public/llama3.1_405b/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl "${DATA_DIR}" -P
    fi
else
    echo "WARNING: Please enter the MLPerf container before downloading dataset"
    echo "WARNING: Llama3.1-405B dataset is NOT downloaded! Exiting..."
    exit 0
fi