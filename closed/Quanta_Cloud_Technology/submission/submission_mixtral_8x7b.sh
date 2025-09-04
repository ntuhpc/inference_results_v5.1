#!/bin/bash

set -ex

MLPINF_DIR=$(dirname $(dirname "$0"))

HF_TOKEN=${HF_TOKEN:-''}
GPU_COUNT=${GPU_COUNT:-'8'}

if [[ "$PREPARE_DATA" == "1" ]]; then
    # Model and Dataset
    bash $MLPINF_DIR/setup/build_model_and_dataset_env.sh

    EXTRA_ARGS="--detach" source $MLPINF_DIR/setup/start_model_and_dataset_env.sh

    # Generate an access token on huggingface and set it here
    docker exec --env HUGGINGFACE_ACCESS_TOKEN="${HF_TOKEN}" ${LAB_DKR_CTNAME} python /lab-mlperf-inference/setup/download_model.py

    docker exec ${LAB_DKR_CTNAME} bash /lab-mlperf-inference/setup/download_mixtral_8x7b.sh

    docker exec ${LAB_DKR_CTNAME} bash /lab-mlperf-inference/setup/quantize_mixtral_8x7b.sh

    docker stop ${LAB_DKR_CTNAME}
fi

MLPERF_IMAGE_NAME=rocm/mlperf-inference:submission_5.1-mixtral_8x7b

if ! docker image inspect "${MLPERF_IMAGE_NAME}" > /dev/null 2>&1; then
    bash $MLPINF_DIR/setup/build_submission_mixtral_8x7b.sh $MLPERF_IMAGE_NAME
fi

bash $MLPINF_DIR/setup/runtime_tunables.sh

EXTRA_ARGS="--detach" source setup/start_submission_env.sh $MLPERF_IMAGE_NAME

# TODO: set options through input variables
docker exec --env COMPANY="TEST" \
            --env CPU_NAME="EPYC_9655" \
            --env GPU_NAME="mi325x" \
            --env GPU_COUNT=${GPU_COUNT} \
            --env RESULTS="/lab-mlperf-inference/submission/results" \
            --env ENABLE_POWER_SETUP=1 \
            ${LAB_DKR_CTNAME} bash /lab-mlperf-inference/submission/mixtral_8x7b.sh

docker stop ${LAB_DKR_CTNAME}
