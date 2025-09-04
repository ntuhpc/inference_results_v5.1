#!/bin/bash

export RUN_LOGS="${RUN_LOGS:-/workspace}"

export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1

WARMUP=100
START_CORE=0
export NUM_CORES=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
export NUM_NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $3}')
export CORES_PER_NUMA=$(lscpu -b -p=Core,Socket,Node | grep -v '^#' | sort -u | awk -F, '{if ($3==0) print $1}' | wc -l)
if [ "${SCENARIO}" == "Server" ]; then
    if [ "${SYSTEM}" == "1-node-2S-GNR_120C" ]; then
        CPUS_PER_INSTANCE=8
    else
        CPUS_PER_INSTANCE=6
    fi
    BATCH_SIZE=1
else
    CPUS_PER_INSTANCE=3
    BATCH_SIZE=2
fi
export INSTANCES_PER_NUMA=$(( CORES_PER_NUMA / CPUS_PER_INSTANCE ))
export NUM_INSTANCES=$(( INSTANCES_PER_NUMA * NUM_NUMA_NODES ))
${WORKSPACE_DIR}/build/bin/mlperf_runner \
    --scenario ${SCENARIO} \
    --mode ${MODE} \
    --user_conf ${USER_CONF} \
    --model_name ${MODEL} \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_DIR} \
    --num_instance ${NUM_INSTANCES} \
    --start_core ${START_CORE} \
    --warmup_iters ${WARMUP} \
    --cpus_per_instance ${CPUS_PER_INSTANCE} \
    --total_sample_count 24781 \
    --batch_size ${BATCH_SIZE} \
    ${EXCLUDE}

mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${RUN_LOGS}/

if [ "${MODE}" == "Accuracy" ] && [ -e ${RUN_LOGS}/mlperf_log_accuracy.json ]; then
    echo " ==================================="
    echo "         Evaluating Accuracy        "
    echo " ==================================="

    python -u ${ENV_DEPS_DIR}/mlperf_inference/vision/classification_and_detection/tools/accuracy-openimages.py \
    	--mlperf-accuracy-file ${RUN_LOGS}/mlperf_log_accuracy.json \
        --openimages-dir ${DATA_DIR} 2>&1 | tee ${RUN_LOGS}/accuracy.txt
fi

if [ -e "${RUN_LOGS}/mlperf_log_summary.txt" ]; then cat "${RUN_LOGS}/mlperf_log_summary.txt"; fi
