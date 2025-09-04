#!/bin/bash

# Controls workload mode
export SCENARIO="${SCENARIO:-Offline}"
export MODE="${MODE:-Performance}"
export OFFLINE_QPS="${OFFLINE_QPS:-0}"
export SERVER_QPS="${SERVER_QPS:-0}"

# Setting environmental paths
export DATA_DIR=/data
export MODEL_DIR=/model
export LOG_DIR=/logs
export RESULTS_DIR=${LOG_DIR}/results
export DOCUMENTATION_DIR=${LOG_DIR}/documentation
export COMPLIANCE_DIR=${LOG_DIR}/compliance
export COMPLIANCE_SUITE_DIR=/workspace/mlperf_inference/compliance/nvidia
export USER_CONF=user.conf
workload_specific_parameters () {
  export WORKLOAD="rgat"
  export MODEL="rgat"
  export IMPL="pytorch-cpu"
  export COMPLIANCE_TESTS="TEST01"
}

workload_specific_run () {

  export ENV_DEPS_DIR=/workspace/RGAT-env
  # Set HW specific qps settings: either manually-defined, GNR, or [default] EMR.
  export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
  export number_numa=`lscpu | grep "NUMA node(s)" | rev | cut -d' ' -f1 | rev`
  if [ "${OFFLINE_QPS}" != "0" ] || [ "${SERVER_QPS}" != "0" ]; then
      echo "*.Offline.target_qps = ${OFFLINE_QPS}" > /workspace/${USER_CONF}
      echo "*.Server.target_qps = ${SERVER_QPS}" >> /workspace/${USER_CONF}
  elif [ "${number_cores}" == "256" ] ; then
      cp /workspace/systems/${USER_CONF}.GNR_128C /workspace/${USER_CONF}
      export SYSTEM="1-node-2S-GNR_128C"
  elif [ "${number_cores}" == "240" ] ; then
      cp /workspace/systems/${USER_CONF}.GNR_120C /workspace/${USER_CONF}
      export SYSTEM="1-node-2S-GNR_120C"
  elif [ "${number_cores}" == "192" ] ; then
      cp /workspace/systems/${USER_CONF}.GNR_96C /workspace/${USER_CONF}
      export SYSTEM="1-node-2S-GNR_96C"
  elif [ "${number_cores}" == "172" ] ; then
      cp /workspace/systems/${USER_CONF}.GNR_86C /workspace/${USER_CONF}
      export SYSTEM="1-node-2S-GNR_86C"
  else
      cp /workspace/systems/${USER_CONF}.EMR /workspace/${USER_CONF}
      export SYSTEM="1-node-2S-EMR"
  fi
  export USER_CONF=user.conf
  
  num_physical_cores=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l)
  num_numa=$(numactl --hardware|grep available|awk -F' ' '{ print $2 }')
  num_sockets=$(lscpu | grep "Socket(s)" | rev | cut -d' ' -f1 | rev)
  MEM_AVAILABLE=$(free -g | awk '/Mem:/ {print $2}')
  
  if [ $MEM_AVAILABLE -lt 1500 ]; then
      echo "Memory is less than 1500GB, setting NUMA balance"
      export NUM_PROC=1
      export CPUS_PER_PROC=$num_physical_cores
      export WORKERS_PER_PROC=$num_sockets
      export CORE_OFFSET="[0,$((num_physical_cores / 2))]" # first core of each socket
  else
      echo "Memory is greater than 1500GB"

      echo "Detecting first core for each NUMA node..."
      # Get list of NUMA nodes (using sort -un to get unique values)
      numa_nodes=$(numactl --hardware | grep -oP "node \K\d+" | sort -un)
      total_numa=$(echo "$numa_nodes" | wc -l)

      # Create an array to store first cores
      declare -a first_cores

      echo "System has $total_numa NUMA nodes"

      # For each NUMA node, find its first core
      for node in $numa_nodes; do
      # Get first physical core ID for this NUMA node
      first_core=$(lscpu -p=CPU,NODE | grep -v '^#' | awk -v node="$node" -F',' '$2 == node {print $1; exit}')
      first_cores+=($first_core)
      echo "NUMA node $node: first core is $first_core"
      done

      # Format the array for use in your script
      core_offset="["
      for i in "${!first_cores[@]}"; do
      if [ $i -gt 0 ]; then
          core_offset+=","
      fi
      core_offset+="${first_cores[$i]}"
      done
      core_offset+="]"
      echo "Core offset array: $core_offset"

      
      # Set configuration variables
      export NUM_PROC=$num_sockets # 1 process per socket
      export CPUS_PER_PROC=$((num_physical_cores / num_sockets)) # #cores per socket
      export WORKERS_PER_PROC=$((num_numa / num_sockets)) # 1 worker per NUMA node
      export CORE_OFFSET=$core_offset # first core of each NUMA node
  fi

  echo "CORE_OFFSET: ${CORE_OFFSET}"
  echo "NUM_PROC: ${NUM_PROC}"
  echo "WORKERS_PER_PROC: ${WORKERS_PER_PROC}"

  export TMP_DIR=/workspace/output_logs
  if [ "${MODE}" == "Accuracy" ]; then
      echo "Run ${MODEL} (${SCENARIO} Accuracy)."
      bash run_accuracy.sh
      cd ${TMP_DIR}
      mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt /workspace/
  else
      echo "Run ${MODEL} (${SCENARIO} Performance)."
      bash run_offline.sh
      cd ${TMP_DIR}
      echo "moving log files "
      mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt /workspace/
  fi

  rm -rf ${TMP_DIR}
  cd /workspace
}

initialize () {
  if [ -f /workspace/audit.config ]; then
      rm /workspace/audit.config
  fi
  bash run_clean.sh
}

prepare_suplements () {
  # Ensure /logs/systems is populated or abort process.
  export SYSTEMS_DIR=${LOG_DIR}/systems
  mkdir -p ${SYSTEMS_DIR}
  cp /workspace/systems/${SYSTEM}.json ${SYSTEMS_DIR}/

  # Populate /logs/code directory
  export CODE_DIR=${LOG_DIR}/code/${WORKLOAD}/${IMPL}
  mkdir -p ${CODE_DIR}
  cp -r /workspace/README.md ${CODE_DIR}/

  # Populate /logs/measurements directory (No distibution between Offline and Server modes)
  export MEASUREMENTS_DIR=${LOG_DIR}/measurements/${SYSTEM}/${WORKLOAD}
  mkdir -p ${MEASUREMENTS_DIR}/${SCENARIO}
  cp /workspace/measurements.json ${MEASUREMENTS_DIR}/${SCENARIO}/${SYSTEM}.json
  cp /workspace/README.md ${MEASUREMENTS_DIR}/${SCENARIO}/
  cp /workspace/user.conf ${MEASUREMENTS_DIR}/${SCENARIO}/
  # Populate /logs/documentation directory
  mkdir -p ${DOCUMENTATION_DIR}
  cp  /workspace/calibration.md ${DOCUMENTATION_DIR}/
}

workload_specific_parameters

# Setting compliance test list (if applicable)
if [[ "${COMPLIANCE_TESTS}" == *"${MODE}"* ]]; then
    export COMPLIANCE_TESTS="${MODE}"
    export MODE="Compliance"
fi

if [ "${MODE}" == "Performance" ]; then
    initialize
    workload_specific_run
    OUTPUT_DIR=${RESULTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}/performance/run_1
    mkdir -p ${OUTPUT_DIR}
    mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}
    prepare_suplements
elif [ "${MODE}" == "Accuracy" ]; then
    initialize
    workload_specific_run
    OUTPUT_DIR=${RESULTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}/accuracy
    mkdir -p ${OUTPUT_DIR}
    mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt ${OUTPUT_DIR}
elif [ "${MODE}" == "Compliance" ]; then
    for TEST in ${COMPLIANCE_TESTS}; do
        initialize
        echo "Running compliance ${TEST} ..."

        if [ "$TEST" == "TEST01" ]; then
                cp ${COMPLIANCE_SUITE_DIR}/${TEST}/${MODEL}/audit.config .
        else
                cp ${COMPLIANCE_SUITE_DIR}/${TEST}/audit.config .
        fi
        workload_specific_run
        OUTPUT_DIR=${COMPLIANCE_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}/${TEST}/output
        mkdir -p ${OUTPUT_DIR}
        mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}

        RESULTS=${RESULTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
        if ! [ -d ${RESULTS} ]; then
            echo "[ERROR] Compliance run could not be verified due to unspecified or non-existant RESULTS dir: ${RESULTS}"
            exit
        else
            COMPLIANCE_VERIFIED=${COMPLIANCE_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
            python ${COMPLIANCE_SUITE_DIR}/${TEST}/run_verification.py -r ${RESULTS} -c ${OUTPUT_DIR} -o ${COMPLIANCE_VERIFIED}
            rm -r ${OUTPUT_DIR}
        fi
    done
else
    echo "[ERROR] Missing value for MODE. Options: Performance, Accuracy, Compliance"
fi
