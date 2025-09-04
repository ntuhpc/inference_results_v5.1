HOST_VOL ?= ${PWD}
CONTAINER_VOL ?= /work
UID := $(shell id -u `whoami`)
HOSTNAME := $(shell hostname)
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)
UNAME := $(shell whoami)


THOR_BASE_IMAGE_REGISTRY ?= gitlab-master.nvidia.com/mlpinf/mlperf-inference



.PHONY: launch_thor
launch_thor_docker:
	docker run --runtime=nvidia --rm -it -w /work \
		-v $(realpath $(HOST_VOL)):$(CONTAINER_VOL) -v $(realpath ${HOME}):/mnt/${HOME} \
		-v /home/engines:/home/engines \
		--cap-add SYS_ADMIN --cap-add SYS_TIME \
		--shm-size=32gb \
		--ulimit memlock=-1 \
		--device /dev/tegra-soc-hwpm --privileged \
		--security-opt apparmor=unconfined --security-opt seccomp=unconfined --security-opt systempaths=unconfined \
		--name thor_mlperf -h thor_mlperf --add-host thor_mlperf:127.0.0.1 \
		--cpuset-cpus $(shell taskset -c -p $$$$ | awk '{print $$NF}') \
		--user $(UID) --net host --device /dev/fuse \
		${THOR_BASE_IMAGE_REGISTRY}:mlpinf-v5.1-cuda13.0-ubuntu24.04-aarch64-Thor $(DOCKER_COMMAND)
