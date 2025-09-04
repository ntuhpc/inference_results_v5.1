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
export COMPLIANCE_DIR=${LOG_DIR}/compliance
export DOCUMENTATION_DIR=${LOG_DIR}/documentation
export COMPLIANCE_SUITE_DIR=/workspace/inference/compliance/nvidia
export USER_CONF=user.conf

workload_specific_parameters () {
  export WORKLOAD="dlrm-v2-99.9"
  export MODEL="dlrm-v2"
  export IMPL="pytorch-cpu"
  export COMPLIANCE_TESTS="TEST01"
}

workload_specific_run () {
    number_threads=`nproc --all`
    export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
    export number_numa=`lscpu | grep "NUMA node(s)" | rev | cut -d' ' -f1 | rev`
    export NUM_SOCKETS=`grep physical.id /proc/cpuinfo | sort -u | wc -l`
    export CPUS_PER_SOCKET=$((number_cores/NUM_SOCKETS))
    export CPUS_PER_PROCESS=${CPUS_PER_SOCKET}  # which determine how much processes will be used
                                                # process-per-socket = CPUS_PER_SOCKET/CPUS_PER_PROCESS
    export CPUS_PER_INSTANCE=1  # instance-per-process number=CPUS_PER_PROCESS/CPUS_PER_INSTANCE
                                # total-instance = instance-per-process * process-per-socket
    export CPUS_FOR_LOADGEN=1   # number of cpus for loadgen
                                # finally used in our code is max(CPUS_FOR_LOADGEN, left cores for instances)
    export BATCH_SIZE=100
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
    export DTYPE=int8
    export TMP_DIR=/workspace/run_tmp

    if [ "${OFFLINE_QPS}" != "0" ] || [ "${SERVER_QPS}" != "0" ]; then
        echo "*.Offline.target_qps = ${OFFLINE_QPS}" > /workspace/${USER_CONF}
        echo "*.Server.target_qps = ${SERVER_QPS}" >> /workspace/${USER_CONF}
    elif [ "${number_cores}" == "256" ] && [ "${number_numa}" == "6" ]; then
        cp /workspace/systems/${USER_CONF}.GNR_128C ${USER_CONF}
        export SYSTEM="1-node-2S-GNR_128C"
    elif [ "${number_cores}" == "240" ] && [ "${number_numa}" == "6" ]; then
        export CPUS_PER_INSTANCE=2
        export BATCH_SIZE=400
        cp /workspace/systems/${USER_CONF}.GNR_120C ${USER_CONF}
        export SYSTEM="1-node-2S-GNR_120C"
    elif [ "${number_cores}" == "192" ] && [ "${number_numa}" == "2" ]; then
        export CPUS_PER_INSTANCE=2
        cp /workspace/systems/${USER_CONF}.GNR_96C ${USER_CONF}
        export SYSTEM="1-node-2S-GNR_96C"
    elif [ "${number_cores}" == "172" ] && [ "${number_numa}" == "4" ]; then
        export CPUS_PER_INSTANCE=2
        cp /workspace/systems/${USER_CONF}.GNR_86C ${USER_CONF}
        export SYSTEM="1-node-2S-GNR_86C"
    else
        export CPUS_PER_INSTANCE=2
        cp /workspace/systems/${USER_CONF}.EMR ${USER_CONF}
        export SYSTEM="1-node-2S-EMR"
    fi

    if [ "$SCENARIO" == "Server" ]; then
        if [ "${MODE}" == "Accuracy" ]; then
            echo "Run ${MODEL} (${SCENARIO} Accuracy)."
            bash run_main.sh server accuracy ${DTYPE}
            cd ${TMP_DIR}
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt ..
        else
            echo "Run ${MODEL} (${SCENARIO} Performance)."
            bash run_main.sh server ${DTYPE}
	    cd ${TMP_DIR}
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ..
        fi
    else
        if [ "${MODE}" == "Accuracy" ]; then
            echo "Run ${MODEL} (${SCENARIO} Accuracy)."
            bash run_main.sh offline accuracy ${DTYPE}
	    cd ${TMP_DIR}
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt ..
        else
            echo "Run ${MODEL} (${SCENARIO} Performance)."
            bash run_main.sh offline ${DTYPE}
	    cd ${TMP_DIR}
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ..
        fi
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
  cp /workspace/scripts/run_calibration.sh ${MEASUREMENTS_DIR}/${SCENARIO}/

  # Populate /logs/documentation directoryAdd commentMore actions
  mkdir -p ${DOCUMENTATION_DIR}
  cp /workspace/calibration.md ${DOCUMENTATION_DIR}/
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
	    if [ "$TEST" == "TEST01" ]; then
                echo "Running TEST01 Part III..."
                cd ${OUTPUT_DIR}
                bash ${COMPLIANCE_SUITE_DIR}/TEST01/create_accuracy_baseline.sh ${RESULTS}/accuracy/mlperf_log_accuracy.json ${OUTPUT_DIR}/mlperf_log_accuracy.json
                cd /workspace
                COMPLIANCE_ACC_DIR=${COMPLIANCE_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}/${TEST}/accuracy
                python tools/accuracy-dlrm.py --mlperf-accuracy-file ${OUTPUT_DIR}/mlperf_log_accuracy_baseline.json > ${COMPLIANCE_ACC_DIR}/baseline_accuracy.txt
                python tools/accuracy-dlrm.py --mlperf-accuracy-file ${COMPLIANCE_ACC_DIR}/mlperf_log_accuracy.json > ${COMPLIANCE_ACC_DIR}/compliance_accuracy.txt
                echo "Completed TEST01 Part III"
            fi
            rm -r ${OUTPUT_DIR}
        fi
    done
else
    echo "[ERROR] Missing value for MODE. Options: Performance, Accuracy, Compliance"
fi
