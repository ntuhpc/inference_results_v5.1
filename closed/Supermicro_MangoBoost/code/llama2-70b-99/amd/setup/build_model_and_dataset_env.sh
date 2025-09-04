#!/bin/bash
set -e

SETUP_DIR=$(dirname -- $0)
TOOLS_DIR=${SETUP_DIR}/../tools
SCRIPTS_DIR=${SETUP_DIR}/dataset_and_model
BASE_IMAGE_NAME=rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0
MLPERF_IMAGE_NAME=mlperf_inference_submission_model_and_dataset_prep:5.0

docker build --no-cache --build-arg BASE_IMAGE=${BASE_IMAGE_NAME} --build-arg TOOLS_DIR=${TOOLS_DIR} --build-arg SCRIPTS_DIR=${SCRIPTS_DIR} -t ${MLPERF_IMAGE_NAME} -f "$SETUP_DIR/Dockerfile.model_and_dataset" .

