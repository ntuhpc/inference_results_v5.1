#!/bin/bash
set -e

SETUP_DIR=$(dirname -- $0)
ARCH=${ARCH:-$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')}
VLLM_IMAGE_ORIG=rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.9.1_mi35X_prealpha2
VLLM_IMAGE_SQUASH=${VLLM_IMAGE_ORIG}-squashed
VLLM_IMAGE_NAME=${VLLM_IMAGE_SQUASH}-fixed
PATCHES="--apply-vllm-patches"
if [[ "$ARCH" == "gfx942" ]]; then
    echo "Unsupported architecture: $ARCH"
    exit 1
fi
docker run -d --name temp_container $VLLM_IMAGE_ORIG /bin/true
docker export temp_container | docker import - $VLLM_IMAGE_SQUASH
docker rm temp_container
docker run -d --name temp_container $VLLM_IMAGE_SQUASH /bin/true
docker commit -c 'CMD ["/bin/bash"]' -c 'ENV PYTORCH_ROCM_ARCH=gfx942;gfx950 ROCM_PATH=/opt/rocm PATH=/opt/ompi/bin:/opt/ucx/bin:/opt/cache/bin:/opt/rocm/llvm/bin:/opt/rocm/opencl/bin:/opt/rocm/hip/bin:/opt/rocm/hcc/bin:/opt/rocm/bin:/opt/conda/envs/py_3.10/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LD_LIBRARY_PATH=/opt/ompi/lib:/opt/rocm/lib:/usr/local/lib:' temp_container $VLLM_IMAGE_NAME
docker rm temp_container
# Create llama3-405b docker
bash $SETUP_DIR/build_docker.sh --base-docker-image $VLLM_IMAGE_NAME --arch $ARCH $PATCHES --custom-vllm-branch 9f6b92db47c344
image_name=$(cat mlperf_image_name.txt)
if [ ! -z "$1" ]; then docker tag $image_name $1; fi
