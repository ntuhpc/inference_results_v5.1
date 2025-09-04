#!/bin/bash
set -xeu

SUBMISSION_DIR=/lab-mlperf-inference/submission
SUBMISSION_PACKAGE_NAME=inference_results_5.1
CPU_NAME=${CPU_NAME:-"EPYC_9554"} #lscpu | grep name
GPU_NAME=${GPU_NAME:-'mi300x'}
GPU_COUNT=${GPU_COUNT:-8}
COMPANY=${COMPANY:-'AMD'}
RESULTS=${RESULTS:-'results'}

# Check if GPU_COUNT is either 1 or 8
if [[ "$GPU_COUNT" -ne 1 && "$GPU_COUNT" -ne 8 ]]; then
   echo "Error: GPU_COUNT must be set to 1 or 8. Current value: $GPU_COUNT"
   exit 1
fi

if [ -f 'audit.config' ]; then
   rm audit.config
fi

CPU_COUNT=$(lscpu | grep 'Socket(s)' | sed 's/[^0-9]*//g')
GPU_NAME_UPPER=$(echo "$GPU_NAME" | sed 's/.*/\U&/')
SYSTEM_NAME="${GPU_COUNT}x${GPU_NAME_UPPER}_${CPU_COUNT}x${CPU_NAME}" #8xMI300X_2xEPYC_9554

SYSTEM_JSON="${GPU_NAME}_system.json"
if [ ! -f "$SYSTEM_JSON" ]; then
   SYSTEM_JSON="dummy_system.json"
fi

function user_conf() {
   model=${1}

   # Get user.conf file path based on the GPU name and GPU count
    USER_CONF_PATH="/lab-mlperf-inference/code/user_${GPU_NAME}"

   if [[ "$GPU_COUNT" -eq 1 ]]; then
      USER_CONF_PATH+=_1gpu
   fi

   USER_CONF_PATH+=.conf
   echo "$USER_CONF_PATH"
}

function package() {
    model=${1}
    suffix=${2:-""}
    scenarios="Server Offline"

    if [ "$model" == "llama2-70b" ]; then
        scenarios="${scenarios} Interactive"
    fi

    # Pick the right user.conf based on the model.
    user_conf_path=$(user_conf $model)

    if [ -d "${RESULTS}/${model}" ]; then
      python $SUBMISSION_DIR/package_submission.py \
         --base-package-dir $SUBMISSION_PACKAGE_NAME \
         --input-dir ${RESULTS}/${model} \
         --scenarios ${scenarios} \
         --system-name ${SYSTEM_NAME} \
         --benchmark ${model}${suffix} \
         --user-conf ${user_conf_path} \
         --company ${COMPANY} \
         --system-json ${SYSTEM_JSON}
   else
      echo "Skipping ${model}, result missing."
   fi
}

package llama2-70b -99
package llama2-70b -99.9
package mixtral-8x7b

zip -r $SUBMISSION_PACKAGE_NAME.zip $SUBMISSION_PACKAGE_NAME
