#!/bin/bash
set -e

SETUP_DIR=$(dirname -- $0)
ARCH=${ARCH:-$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')}
VLLM_IMAGE_NAME=rocm/vllm-dev:nightly_0610_rc2_0610_rc2_20250605
PATCHES="--apply-vllm-patches --apply-llama2-patches"
if [[ "$ARCH" == "gfx950" ]]; then
    VLLM_IMAGE_NAME=rocm/pytorch-private:vllm_llama405b_rocm7.0_0627_ck_decode
    PATCHES="$PATCHES --apply-aiter-patches"
fi
# Create llama2-70b docker
bash $SETUP_DIR/build_docker.sh --base-docker-image $VLLM_IMAGE_NAME --arch $ARCH $PATCHES
image_name=$(cat mlperf_image_name.txt)
if [ ! -z "$1" ]; then docker tag $image_name $1; fi
