#!/bin/bash

CHECKSUMS=(
            /model/retinanet-model.pth,a55f6bec3464f605ce8d686da8ac1533
          )

echo "Downloading model..."
export MODEL_DIR=/model
wget --no-check-certificate 'https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth' -O "${MODEL_DIR}/retinanet-model.pth"

echo "Begining validation..."
cd /workspace
for ITEM in "${CHECKSUMS[@]}"; do
    FILENAME=$(echo ${ITEM} | cut -d',' -f1)
    CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
    bash run_validation.sh ${FILENAME} ${CHECKSUM}
done
