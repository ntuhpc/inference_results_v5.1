#!/bin/bash

srun --container-image /mnt/data/shobhitv/13.0-devel.mlperf-skinny.trtllm-dsr1-again.sqsh --nodes=1 --ntasks-per-node=1 --mpi=pmi2 python3 -c "import torch"
srun --container-image /mnt/data/shobhitv/13.0-devel.mlperf-skinny.trtllm-dsr1-again.sqsh --nodes=1 --ntasks-per-node=1 --mpi=pmi2 trtllm-serve --help
