#!/bin/bash
set -xeu

DOCKER_IMNAME=rocm/pytorch-private:vllm_llama405b_rocm7.0_0627_ck_decode


docker run -it -d --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --name=mlperf_finetune \
        -v /mlperf_finetune/LLaMA-Factory/:/LLaMA-Factory/ \
        -v /mlperf_finetune/data/:/data/ \
        -v /mlperf_finetune/model/:/model/ \
        -v /mlperf_finetune/quantization/:/quantization/ \
        ${DOCKER_IMNAME}
