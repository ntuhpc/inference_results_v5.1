#!/bin/bash
set -e

SETUP_DIR=$(dirname -- $0)
VLLM_IMAGE_NAME=rocm/vllm-dev:nightly_0610_rc2_0610_rc2_20250605
# Create llama2-70b docker
bash $SETUP_DIR/build_docker.sh --base-docker-image $VLLM_IMAGE_NAME --apply-vllm-patches --apply-llama2-patches
image_name=$(cat mlperf_image_name.txt)
if [ ! -z "$1" ]; then docker tag $image_name $1; fi
