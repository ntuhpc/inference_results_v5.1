#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
export WORKLOAD="retinanet"
export MODEL="retinanet"
export IMPL="pytorch-cpu"
export COMPLIANCE_TESTS="TEST01"
export COMPLIANCE_SUITE_DIR=/workspace/retinanet-env/mlperf_inference/compliance/nvidia
export MAX_LATENCY=100000000

# This function should handle each combination of the following parameters:
# - SCENARIO: Offline or Server
# - MODE: Performance, Accuracy, and Compliance
workload_specific_run () {
  export SCENARIO=${SCENARIO}
  export MODE=${MODE}

  # Standard ENV settings (potentially redundant)
  export DATA_DIR=${DATA_DIR}
  export USER_CONF=${USER_CONF}
  export RUN_LOGS=${RUN_LOGS}

  # Workload run-specific settings
  export ENV_DEPS_DIR=/workspace/retinanet-env
  export MODEL_CHECKPOINT=/model/retinanet-model.pth
  export MODEL_PATH=/model/retinanet-int8-model.pth

  if [ "${MODE}" == "Compliance" ]; then
    export MODE="Performance"
  fi
  cd ${WORKSPACE_DIR}

  # Following v5.0 Inference release, one file handles all scenarios/modes: 
  bash run_local.sh
}
