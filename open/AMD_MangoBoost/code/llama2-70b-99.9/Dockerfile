# Copyright 2025, MangoBoost, Inc. All rights reserved.

# Declare build-time args early
ARG AMD_BASE_IMAGE=default
ARG LLMBOOST_BASE_IMAGE=llmboost/mb-llmboost:mlperf-5_1-multi-node-mi35x-base

# Stage 1: Source stage to copy from LLMBoost
FROM ${LLMBOOST_BASE_IMAGE} AS llmboost

# Stage 2: AMD ROCm base image
FROM ${AMD_BASE_IMAGE} AS base

# Stage 3: Final runtime image based on AMD ROCm
FROM base AS final

# Re-declare ARGs
ARG PLATFORM

# Set CUDA_DEVICE_ORDER only if PLATFORM is set
RUN if [ -n "$PLATFORM" ]; then \
    echo "export CUDA_DEVICE_ORDER=PCI_BUS_ID" >> /etc/profile.d/llmboost.sh; \
    fi

# Now it's safe to copy from the earlier llmboost stage
COPY --from=llmboost /workspace /workspace
COPY --from=llmboost /opt/mangoboost /opt/mangoboost
COPY --from=llmboost /lab-mlperf-inference /lab-mlperf-inference

# use workspace as the working directory
WORKDIR /workspace
ENTRYPOINT ["/opt/mangoboost/entrypoint.sh"]

# Set environment variables 
ENV HIP_FORCE_DEV_KERNARG=1 \
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1 \
    VLLM_TUNE_GEMM=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    # Make sure /pypath exists and drop in our customizer
    PYTHONPATH=/pypath/.config:/workspace/ \
    CUDA_DEVICE_ORDER=${PLATFORM:+PCI_BUS_ID} \
    HAYSTACK_AUTO_TRACE_ENABLED=false \
    HAYSTACK_TELEMETRY_ENABLED=false \
    OTEL_TRACES_EXPORTER=none

# Install necessary packages and Python dependencies
RUN apt update --allow-releaseinfo-change \
    && apt install -y numactl zip unzip libsqlcipher-dev tree\
    && rm -rf /var/lib/apt/lists/* \
    # Install Loadgen
    && CFLAGS="-std=c++14 -O3" python3 -m pip install git+https://github.com/mlcommons/inference#subdirectory=loadgen && \
    # Install Packages
    python3 -m pip install --upgrade pip && \
    pip install pysqlcipher3 && \
        (dpkg -s python3-blinker >/dev/null 2>&1 && apt-get remove -y python3-blinker || true) && \
        python3 -m pip install --upgrade --break-system-packages \
        fastapi absl-py nltk evaluate rouge_score py-libnuma \
        websockets faster-fifo gunicorn accelerate uvloop \
        pydantic bcrypt 'httpx[http2]' haystack-ai milvus-haystack \
        sentence-transformers librosa matplotlib flask hydra-core;

# Install LLMBoost
RUN python3 -m pip install /workspace/*.whl && \
    rm -rf /workspace/*.whl;
