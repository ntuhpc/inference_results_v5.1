# syntax=docker/dockerfile:1.4

ARG BASE_IMAGE
FROM ${BASE_IMAGE} as base

ARG ENV
ARG BUILD_CONTEXT
ARG CUDA_VER
ARG TRT_MAJOR_VER
ARG TRT_VER
ARG USE_NIGHTLY
ARG MITTEN_HASH
ARG INSTALL_RGAT_DEPS

ENV ENV=${ENV}
ENV BUILD_CONTEXT=${BUILD_CONTEXT}
ENV CUDA_VER=${CUDA_VER}
ENV TRT_MAJOR_VER=${TRT_MAJOR_VER}
ENV TRT_VER=${TRT_VER}
ENV USE_NIGHTLY=${USE_NIGHTLY}
ENV MITTEN_GIT_URL=ssh://git@gitlab-master.nvidia.com:12051/mlpinf/mitten.git
ENV MITTEN_HASH=${MITTEN_HASH}
ENV INSTALL_RGAT_DEPS=${INSTALL_RGAT_DEPS}

SHELL ["/bin/bash", "-c"]

WORKDIR /tmp

# 1. apt installs
COPY --from=common /install_apt_deps.sh .
RUN if [ -f install_apt_deps.sh ]; then bash install_apt_deps.sh; fi

# 2. pip installs
COPY --from=common /requirements.${BUILD_CONTEXT}.* .
COPY --from=common /install_pip_deps.sh .
RUN if [ -f install_pip_deps.sh ]; then bash install_pip_deps.sh; fi

# 3. Any from-source build required.
COPY --from=common /install_misc_deps.sh .
RUN if [ -f install_misc_deps.sh ]; then bash install_misc_deps.sh; fi

# 4. install docker-within-docker
COPY --from=common /install_docker.sh .
RUN if [ -f install_docker.sh ]; then bash install_docker.sh; fi

# 5. install TensorRT for dev mode
COPY install_tensorrt.sh .
RUN if [ -f install_tensorrt.sh ]; then bash install_tensorrt.sh; fi

# 6. install mlperf mitten
RUN --mount=type=ssh GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" \
    git clone ${MITTEN_GIT_URL} /tmp/mitten
COPY --from=common /install_mitten.sh .
RUN if [ -f install_mitten.sh ]; then bash install_mitten.sh; fi

# 6. RGAT installs
COPY --from=common patches/ /tmp/patches
COPY --from=common install_rgat_deps.sh .
RUN if [ "$INSTALL_RGAT_DEPS" = "1" ]; then if [ -f install_rgat_deps.sh ]; then bash install_rgat_deps.sh; fi; fi

WORKDIR /work
