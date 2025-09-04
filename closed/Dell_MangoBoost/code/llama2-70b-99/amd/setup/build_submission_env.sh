#!/bin/bash
set -e

SETUP_DIR=$(dirname -- $0)
VLLM_IMAGE_NAME=rocm/vllm-dev:nightly_main_20250203_hipblaslt_6b6a724
MLPERF_IMAGE_NAME=mlperf_inference_submission:5.0-base
MLPERF_FINAL_IMAGE_NAME=mlperf_inference_submission:5.0

docker build --no-cache --build-arg BASE_IMAGE=${VLLM_IMAGE_NAME} -f "$SETUP_DIR/Dockerfile.mlperf" -t ${MLPERF_IMAGE_NAME} "$SETUP_DIR/.."
docker build --no-cache --build-arg BASE_IMAGE=${MLPERF_IMAGE_NAME} --build-arg VLLM_DIR="$SETUP_DIR/vllm" -f "$SETUP_DIR/Dockerfile.vllm" -t ${MLPERF_FINAL_IMAGE_NAME} "$SETUP_DIR/.."

# Update and enable it when we get specific branched
# MLPERF_TUNED_IMAGE_NAME=mlperf_inference_submission:5.0-tuned
# export HIPBLASLT_BRANCH="4d40e36"
# export HIPBLAS_COMMON_BRANCH="7c1566b"
# export LEGACY_HIPBLASLT_OPTION=""
# docker build --no-cache --build-arg BASE_IMAGE=${MLPERF_FINAL_IMAGE_NAME} --build-arg HIPBLASLT_BRANCH=${HIPBLASLT_BRANCH} --build-arg HIPBLAS_COMMON_BRANCH=${HIPBLAS_COMMON_BRANCH} --build-arg LEGACY_HIPBLASLT_OPTION=${LEGACY_HIPBLASLT_OPTION} -f "$SETUP_DIR/Dockerfile.hipblaslt" -t ${MLPERF_TUNED_IMAGE_NAME} "$SETUP_DIR/.."
