#!/bin/bash
#SBATCH --output=logs/run_%j/stdout.txt

repo_root=$(git rev-parse --show-toplevel)

# Cross node parallelism for ds-r1
## This script will launch $num_nodes server instances, each on $num_nodes_dp1 nodes.

num_nodes_dp1=2
num_tasks_per_node=4

num_nodes=$SLURM_JOB_NUM_NODES
num_total_gpus=$((num_nodes * num_tasks_per_node))

num_servers=$((num_nodes / num_nodes_dp1))

# make a temp directory ./run_xxxx
dir_name="build/slurm_logs/run_$SLURM_JOB_ID"
mkdir -p $dir_name


usage="sbatch \\
    run_server.sh \\
    --mlperf_container_image=/path/to/mlperf/sqsh \\
    --mlperf_scratch_path=/path/to/mlperf_inference_storage \\
    --trt_engine_artefacts=/path/to/large/vol/storage \\
    --scenario=mlperf_scenario \\
    --benchmark_name=deepseek-r1 \\
    --core_type=trtllm_endpoint \\
    --gpus_per_instance=num_gpus_per_model \\
    --system_name=SYSTEM_NAME \\
    --run_client=1"

system_name=""
run_client=0
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mlperf_container_image=*)
            mlperf_container_image="${1#*=}"
            shift
            ;;
        --mlperf_scratch_path=*)
            mlperf_scratch_path="${1#*=}"
            shift
            ;;
        --trt_engine_artefacts=*)
            trt_engine_artefacts="${1#*=}"
            shift
            ;;
        --scenario=*)
            scenario="${1#*=}"
            shift
            ;;
        --benchmark_name=*)
            benchmark_name="${1#*=}"
            shift
            ;;
        --core_type=*)
            core_type="${1#*=}"
            shift
            ;;
        --gpus_per_instance=*)
            gpus_per_instance="${1#*=}"
            shift
            ;;
        --system_name=*)
            system_name="${1#*=}"
            shift
            ;;
        --run_client=*)
            run_client="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $usage"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $usage"
            exit 1
            ;;
    esac
done

if [ -z "$mlperf_container_image" ]; then
    echo "Error: --mlperf_container_image is not provided"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$mlperf_scratch_path" ]; then
    echo "Error: --mlperf_scratch_path is not provided"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$trt_engine_artefacts" ]; then
    echo "Error: --trt_engine_artefacts is not provided"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$scenario" ]; then
    echo "Error: --scenario is not specified"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$benchmark_name" ]; then
    echo "Error: --benchmark_name is not specified"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$core_type" ]; then
    echo "Error: --core_type is not specified"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$gpus_per_instance" ]; then
    echo "Error: --gpus_per_instance is not specified"
    echo "Usage: $usage"
    exit 1
fi

num_server_instances=$((num_total_gpus / gpus_per_instance))
num_nodes_per_server=$((num_nodes / num_server_instances))
echo "Will spawn $num_server_instances server instances, each on $num_nodes_per_server nodes"

# Export variables that will be used in srun commands
export mlperf_container_image
export mlperf_scratch_path
export trt_engine_artefacts
export scenario
export benchmark_name
export core_type

export container_workdir="/work"
export script_dir="$container_workdir/scripts/slurm_llm"
export actual_workdir="${repo_root}/closed/NVIDIA"

export server_container_name="mlperf_inference"
export container_mount="$actual_workdir:$container_workdir,$mlperf_scratch_path:/home/mlperf_inference_storage,$trt_engine_artefacts:/home/artefacts"

set -x

export RUN_ARGS="--benchmarks=$benchmark_name \
 --scenarios=$scenario \
 --core_type=$core_type \
 --trtllm_server_urls=0.0.0.0:30000 \
 --server_in_foreground"

export server_srun_header="srun --container-image=$mlperf_container_image \
 --container-mounts=$container_mount \
 --container-workdir=$container_workdir \
 --container-remap-root \
 --export=RUN_ARGS,script_dir,SYSTEM_NAME=$system_name \
 --mpi=pmi2"

# make run_llm_server
for i in $(seq 1 $num_server_instances); do
    $server_srun_header \
    --container-name=mlperf_inference-run_llm_server \
    --nodes=$num_nodes_per_server \
    --ntasks=$gpus_per_instance \
    --output=$dir_name/server-launch-log-$i.txt \
    /bin/bash -c 'hostname && source $script_dir/cross_node_instances/prefix.sh && make run_llm_server' &
done

format_hostnames() {
    local num_servers=$1
    shift
    local hosts=("$@")
    local result=""

    for ((i=0; i<num_servers; i++)); do
        if [[ -n "$result" ]]; then
            result+=","
        fi
        result+="${hosts[$((i*num_nodes_dp1))]}:30000"
    done

    echo "$result"
}

node_list=$(scontrol show hostnames $SLURM_NODELIST)
endpoints=$(format_hostnames $num_servers $node_list)
echo "trtllm_server_urls: $endpoints"

if [ "$run_client" = 1 ]; then
    # warmup and health check is disabled, hence wait for server
    sleep 240
    export RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=$scenario --trtllm_server_urls=${endpoints} --trtllm_runtime_flags=max_concurrency:$concurrency"
    export SYSTEM_NAME="GB300-NVL${num_total_gpus}"
    srun --overlap --nodes=1 --ntasks=1 \
        --container-image=${mlperf_container_image} \
        --mpi=pmi2 \
        --container-mounts=/project/coreai_mlperf_inference/shobhitv/mlpinf-repos/mlperf-inference/closed/NVIDIA:/work,/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone/:/home/mlperf_inference_storage \
        --export=RUN_ARGS,SYSTEM_NAME \
        --container-workdir=/work \
        --output=${dir_name}/mlperf-harness-run.out \
        /bin/bash -c "pip install build/inference/loadgen/mlcommons_loadgen*.whl orjson && make run_harness"

    srun --overlap --ntasks-per-node=1 \
        --container-name=mlperf_inference-run_llm_server \
        pkill -9 trtllm-serve
else
    wait
fi

