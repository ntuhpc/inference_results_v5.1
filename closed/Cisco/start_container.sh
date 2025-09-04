#!/bin/bash

export LLAMA405B_DATADIR="<LLAMA405B_DATADIR>"
export LLAMA405B_MODELDIR="<LLAMA405B_MODELDIR>"
export LLAMA405B_PREPROC_DATADIR="<LLAMA405B_PREPROC_DATADIR>"

export LLAMA70B_DATADIR="<LLAMA70B_DATADIR>"
export LLAMA70B_MODELDIR="</mnt/vast/mlperf_data/models/Llama-2-70b-chat-hf/>"

export ENGINES_DIR="</mnt/vast/mlperf_data/models/engines/>"

export CONT_NAME="<CONT_NAME>"

docker run --gpus=all --runtime=nvidia --rm -it -w /work \
        --cap-add SYS_ADMIN --cap-add SYS_TIME \
        -v $SSH_AUTH_SOCK:$SSH_AUTH_SOCK \
        -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK \
        --ipc host \
	-e NVIDIA_VISIBLE_DEVICES=all \
        --shm-size=32gb \
        --ulimit memlock=-1 \
        -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
        --security-opt apparmor=unconfined --security-opt seccomp=unconfined \
        --name mlperf-inference-${HOSTNAME}-x86_64 -h mlperf-inference-${HOSTNAME}-x86-64 --add-host mlperf-inference-${HOSTNAME}-x86_64:127.0.0.1 \
        --cpuset-cpus 0-255 \
        --user 1000 --net host --device /dev/fuse \
	-v $LLAMA405B_DATADIR:/work/build/data/llama3.1-405b/ \
        -v $LLAMA405B_PREPROC_DATADIR:/data/preprocessed_data/llama3.1-405b/ \
	-v $LLAMA405B_MODELDIR:/work/build/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct \
        -v /mnt/vast/mlperf_data/Llama3.1-405B/:/work/build/models/Llama3.1-405B/ \
	-v $LLAMA70B_DATADIR:/data/llama2-70b/  \
	-v $LLAMA70B_MODELDIR:/work/build/models/Llama2/Llama-2-70b-chat-hf/ \
	-v $ENGINES_DIR:/data/engines/ \
 	-e MLPERF_SCRATCH_PATH=/work/build/ \
        $CONT_NAME


