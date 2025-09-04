#!/bin/bash

CHECKSUMS=(
           /data/annotations,7af8ef52d3f3a5130c869a197a063d07
	   /data/validation,d934733e88783445734ebe81400c5d02
          )

echo "Downloading dataset..."
export DATA_DIR=/data
cd /workspace/retinanet-env/mlperf_inference/vision/classification_and_detection/tools
bash openimages_mlperf.sh --dataset-path ${DATA_DIR}

echo "Begining validation..."
cd /workspace
for ITEM in "${CHECKSUMS[@]}"; do
    FILENAME=$(echo ${ITEM} | cut -d',' -f1)
    CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
    bash run_validation.sh ${FILENAME} ${CHECKSUM}
done
