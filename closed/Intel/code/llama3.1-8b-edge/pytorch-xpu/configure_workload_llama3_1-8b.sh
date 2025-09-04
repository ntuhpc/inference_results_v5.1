#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
export WORKLOAD="llama3.1-8b"
export MODEL="llama3_1-8b"
export IMPL="pytorch-xpu"
export COMPLIANCE_TESTS="TEST06"
export COMPLIANCE_SUITE_DIR=${WORKSPACE_DIR}/mlperf-inference/compliance/nvidia
#export MAX_LATENCY=10000000000

# This function should handle each combination of the following parameters:
# - SCENARIO: Offline or Server
# - MODE: Performance, Accuracy, and Compliance
workload_specific_run () {
  rm -rf /workspace/log
  rm -rf /workspace/run_output
  mkdir -p /workspace/run_output
  
  if [ "$SCENARIO" == "Server" ]; then
      source init_env.sh llama3_1-8b server
      if [ "${MODE}" == "Accuracy" ]; then
          echo "Run ${MODEL} (${SCENARIO} Accuracy)."
          bash run_local.sh accuracy
      else
          echo "Run ${MODEL} (${SCENARIO} Performance)."
          bash run_local.sh
      fi
  else
      source init_env.sh llama3_1-8b offline
      if [ "${MODE}" == "Accuracy" ]; then
          echo "Run ${MODEL} (${SCENARIO} Accuracy)."
          bash run_local.sh accuracy
      else
          echo "Run ${MODEL} (${SCENARIO} Performance)."
          bash run_local.sh
      fi
  fi

  mv log/llama/* /workspace/run_output/
}
