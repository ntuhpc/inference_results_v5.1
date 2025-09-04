#!/bin/bash

CHECKSUMS=(
           /data/day_23_dense.npy,cdf7af87cbc7e9b468c0be46b1767601
	   /data/day_23_labels.npy,dd68f93301812026ed6f58dfb0757fa7
	   /data/day_23_sparse_multi_hot.npz,c46b7e31ec6f2f8768fa60bdfc0f6e40
          )
	  
echo "Downloading dataset..."
export DATA_DIR=/data
cm run script --tags=get,preprocessed,dataset,criteo,_multihot,_mlc  -j --to=${DATA_DIR}

# cm doesn't recognize 'to' location.  Manually finding and moving from cache.
DLRM_PREPROCESSED=$(find /root/CM/repos/local/cache -name dlrm_preprocessed)
mv ${DLRM_PREPROCESSED}/* /data/

echo "Begining validation..."
cd /workspace
for ITEM in "${CHECKSUMS[@]}"; do
    FILENAME=$(echo ${ITEM} | cut -d',' -f1)
    CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
    bash run_validation.sh ${FILENAME} ${CHECKSUM}
done
