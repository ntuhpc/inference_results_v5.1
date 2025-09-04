#!/bin/bash

CHECKSUMS=(
            /model/model_weights,a4c95e63dfc4229252c2037c698ab4a9
          )

echo "Downloading model..."
export MODEL_DIR=/model
cm run script --tags=get,ml-model,dlrm,_pytorch,_weight_sharded,_rclone -j --to=${MODEL_DIR}

echo "Begining validation..."
cd /workspace
for ITEM in "${CHECKSUMS[@]}"; do
    FILENAME=$(echo ${ITEM} | cut -d',' -f1)
    CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
    bash run_validation.sh ${FILENAME} ${CHECKSUM}
done
