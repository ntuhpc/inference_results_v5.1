#!/bin/bash

export DOCKER_ARGS="<DOCKER_ARGS>"

export UCX_NET_DEVICES="<UCX_NET_DEVICES>"
export NCCL_IB_HCA="<NCCL_IB_HCA>"
export NCCL_SOCKET_IFNAME="<NCCL_SOCKET_IFNAME>"
export NCCL_IB_GID_INDEX=3

export LLAMA405B_DATADIR="<LLAMA405B_DATADIR>"
export LLAMA405B_MODELDIR="<LLAMA405B_MODELDIR>"
export LLAMA405B_PREPROC_DATADIR="<LLAMA405B_PREPROC_DATADIR>"

export LLAMA70B_DATADIR="<LLAMA70B_DATADIR>"
export LLAMA70B_MODELDIR="<LLAMA70B_MODELDIR>"
export ENGINES_DIR="<ENGINES_DIR>"


export CONT_NAME="<CONT_NAME>"
export HOST_NAME="<HOST_NAME>"
export OMPI_MCA_orte_launch_agent="docker run --runtime=nvidia --gpus=all --rm -w /work \
        -e NCCL_IB_HCA=${NCCL_IB_HCA} -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} -e UCX_NET_DEVICES=${UCX_NET_DEVICES} \
        -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} -e NCCL_DEBUG=${NCCL_DEBUG} -e NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS} \
        ${DOCKER_ARGS} \
        -v $SSH_AUTH_SOCK:$SSH_AUTH_SOCK \
        -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK \
        --ipc host \
	--cap-add SYS_ADMIN --cap-add SYS_TIME \
        -e NVIDIA_VISIBLE_DEVICES=all \
        --shm-size=32gb \
        --ulimit memlock=-1 \
        -v $LLAMA405B_DATADIR:/work/build/data/llama3.1-405b/ \
        -v $LLAMA405B_PREPROC_DATADIR:/data/preprocessed_data/llama3.1-405b/ \
	-v $LLAMA405B_MODELDIR:/work/build/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct \
        -v $LLAMA70B_DATADIR:/data/llama2-70b/  \
	-v $LLAMA70B_MODELDIR:/work/build/models/Llama2/Llama-2-70b-chat-hf/ \
	-v $ENGINES_DIR:/data/engines/ \
        -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
	--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
        --cpuset-cpus 0-255 \
	--user 1000 --net host --device /dev/fuse \
        --name $HOST_NAME -h $HOST_NAME \
        -e MLPERF_SCRATCH_PATH=$MLPERF_SCRATCH_PATH \
        -e HOST_HOSTNAME=$HOST_NAME \
        $CONT_BASE_NAME:$CONT_NAME orted"

