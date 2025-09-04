#!/bin/bash

# set -x
set -e

PYTHON3_PATH=/usr/bin/python3
DATASET_FILE_PATH=/data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl
MODEL_PATH=/model/llama3.1-405b/fp8_quantized
EVAL_SCRIPT_PATH=/app/mlperf_inference/language/llama3.1-405b/evaluate-accuracy.py

ARCH=${ARCH:-$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')}
if [[ "$ARCH" == "gfx950" ]]; then
    PYTHON3_PATH=/opt/conda/envs/py_3.10/bin/python
    MODEL_PATH=/model/llama3.1-405b/fp4_quantized/pruned_59_84
fi

if [ ! -f ${PYTHON3_PATH} ]; then
    echo "Wrong python3 path"
    exit 1
fi

if [ ! -f ${DATASET_FILE_PATH} ]; then
    echo "dataset not found, check the README.md how to download it"
    exit 1
fi

if [ ! -d ${MODEL_PATH} ]; then
    echo "model not found, check the README.md how to download it"
    exit 1
elif [ -z "$(ls -A ${MODEL_PATH})" ]; then
    echo "model dir is empty, check the README.md how to download it"
    exit 1
fi

if [ ! -f ${EVAL_SCRIPT_PATH} ]; then
    echo "tools/evaluate-accuracy.py not found"
    exit 1
fi

ACCURACY_JSON=${1}

if [ -z ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    exit 1
fi

if [ ! -f ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    exit 1
fi

OUTPUT_DIR=$(dirname ${ACCURACY_JSON})
RESULT_TXT=${OUTPUT_DIR}/accuracy.txt

${PYTHON3_PATH} -u ${EVAL_SCRIPT_PATH} --checkpoint-path ${MODEL_PATH} \
                                       --mlperf-accuracy-file ${ACCURACY_JSON} \
                                       --dataset-file ${DATASET_FILE_PATH} \
                                       --dtype int32 > ${RESULT_TXT}

echo "Check $RESULT_TXT for the accuracy scores"
