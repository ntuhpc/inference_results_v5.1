#!/bin/bash

source "./environment_var_resolver.sh"
source "llama_configs/config_mpi_launcher.sh"

CURRENT_DATE=$(date +"%Y.%m.%d-%H.%M.%S")

export LOG_DIR_BASE="/work/build/logs/${CURRENT_DATE}"
export PRINT_VARS=1

if [ "$PRINT_VARS" == "1" ]; then
  echo "NUM_GPUS: $NUM_GPUS"
  echo "SCENARIO: $SCENARIO"
  echo "TEST_MODE: $TEST_MODE"
  echo "MODEL: $MODEL"
  echo "GPU_TYPE: $GPU_TYPE"
  echo "ENGINE_PATH: $ENGINE_PATH"
  echo "BATCH_SIZE: $BATCH_SIZE"
  echo "MLPERF_CONF_PATH: $MLPERF_CONF_PATH"
  echo "USER_CONF_PATH: $USER_CONF_PATH"
  echo "LOG_OUTPUT: $LOG_OUTPUT"
  echo "CONT_BASE_NAME: $CONT_BASE_NAME"
  echo "HOSTFILE: $HOSTFILE"
fi

mkdir -p "$LOG_OUTPUT"

procs=$((NUM_GPUS+1))

(
mpirun --allow-run-as-root \
      --wd /work \
      --display-map \
      --tag-output \
      --bind-to numa  \
      -x NCCL_SOCKET_IFNAME \
      -x UCX_NET_DEVICES \
      -x NCCL_IB_GID_INDEX \
      -x NCCL_IB_HCA \
      -x NCCL_DEBUG=WARN \
      -np $procs --rf $RANKFILE \
    python3 -m code.harness.harness_llm_py.runner_multinode \
    --logfile_outdir="$LOG_OUTPUT" \
    --logfile_prefix="mlperf_log_" \
    --performance_sample_count="$PERFORMANCE_SAMPLE_COUNT" \
    --test_mode="$TEST_MODE" \
    --gpu_batch_size=$BATCH_SIZE \
    --tensor_path="$TENSOR_PATH" \
    --use_graphs=false \
    --use_token_latencies=true \
    --enable_sort=false \
    --llm_gen_config_path="$LLM_GEN_CONFIG_PATH" \
    --trtllm_checkpoint_flags="kv_cache_dtype:fp8" \
    --trtllm_build_flags="$TRTLLM_BUILD_FLAGS" \
    --trtllm_runtime_flags="$TRTLLM_RUNTIME_FLAGS" \
    --gpu_engines="$ENGINE_PATH" \
    --mlperf_conf_path="$MLPERF_CONF_PATH" \
    --user_conf_path="$USER_CONF_PATH" \
    --scenario "$SCENARIO" \
    --model "$MODEL" \
    --dataset_cls "$DATASET_CLS"
) 2>&1 | tee /work/logs/output_${NUM_GPUS}x${GPU_TYPE}_${MODEL}_${SCENARIO}_${TEST_MODE}_.log

echo "Test completed. Logs are available in /work/logs/output_${NUM_GPUS}x${GPU_TYPE}_${MODEL}_${SCENARIO}_${TEST_MODE}_.log"

if [ "$TEST_MODE" = "AccuracyOnly" ]; then
  
  if [ "$MODEL" = "llama2-70b" ]; then
    python3 scripts/evaluate_accuracy_70b.py --checkpoint-path ${HUGGINGFACE_CHECKPOINT} \
        --mlperf-accuracy-file ${LOG_OUTPUT}/mlperf_log_accuracy.json \
        --dataset-file ${DATASET_PATH} \
        --dtype int32 >&1 | tee ${LOG_OUTPUT}/accuracy.txt
  else
    python3 scripts/evaluate_accuracy_405b.py --checkpoint-path ${HUGGINGFACE_CHECKPOINT} \
        --mlperf-accuracy-file ${LOG_OUTPUT}/mlperf_log_accuracy.json \
        --dataset-file ${DATASET_PATH} \
        --dtype int32 >&1 | tee ${LOG_OUTPUT}/accuracy.txt
  fi
fi

