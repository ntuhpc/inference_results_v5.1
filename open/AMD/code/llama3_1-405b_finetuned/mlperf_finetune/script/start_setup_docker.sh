#!/bin/bash
set -xeu

DOCKER_IMNAME=rocm/pytorch-private:vllm-fp4-405b-250603-asm-rc2


docker run -it -d --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --name=mlperf_setup \
        -v /mlperf_finetune/script/:/script/ \
        -v /mlperf_finetune/data/:/data/ \
        -v /mlperf_finetune/model/:/model/ \
        -v /mlperf_finetune/RULER/:/RULER/ \
        -v /mlperf_finetune/quantization/:/quantization/ \
        ${DOCKER_IMNAME}
