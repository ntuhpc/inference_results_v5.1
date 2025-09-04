# Llama2-70B Inference on Intel Arc Pro B60

## LEGAL DISCLAIMER
To the extent that any data, datasets, or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality. By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license.

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets, or models.

## Set the Host OS Directories
Set the directories on the host system where model, dataset, and log files will reside. These locations will retain model and data content between Docker sessions.
```
export DATA_DIR="${DATA_DIR:-${PWD}/data}"
export MODEL_DIR="${MODEL_DIR:-${PWD}/model}"
export LOG_DIR="${LOG_DIR:-${PWD}/logs}"
```

## Launch the Docker Image
In the Host OS environment, run the following after setting the proper Docker image name. If the Docker image is not on the system already, it will be retrieved from the registry.

If retrieving the model or dataset, ensure any necessary proxy settings are run inside the container.
```
export DOCKER_IMAGE=intel/intel-optimized-pytorch:mlperf-inference-5.1-llama2-70b_xpu

docker run --privileged -it --rm \
        --ipc=host --net=host --cap-add=ALL \
        -e http_proxy=${http_proxy} \
        -e https_proxy=${https_proxy} \
        -v ${DATA_DIR}:/data \
        -v ${MODEL_DIR}:/model \
        -v ${LOG_DIR}:/logs \
        --workdir /workspace \
        ${DOCKER_IMAGE} /bin/bash
```

## Download Model and Dataset [one-time operation]
From inside the Docker container, follow the MLCommons instructions for downloading the model: [Link](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md#get-model).

From inside the Docker container, follow the MLCommons instructions for downloading the dataset: [Link](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md#get-dataset). Ensure the dataset files reside directly within the following directory (relative to the launched container):
```
/data/open_orca
```

## Calibrate the Model [one-time operations]
Run this step inside the Docker container.  This operation will create and preserve a calibrated model along with the original model file.
```
bash calibration/quantize_8b.sh
```

## Initialize the Environment
Run this step inside the Docker container.  This operation must be completed at the launch of each Docker session (not each run).
```
bash initialize.sh
```

## Run Benchmark (submission)
Run this step inside the Docker container.  Select the appropriate scenario.  If this is the first time running this workload, the original model file will be calibrated to INT8 and stored alongside the original model file (one-time operation).

Performance::
```
SCENARIO=Offline MODE=Performance bash run_mlperf_llama3_1-8b.sh
SCENARIO=Server  MODE=Performance bash run_mlperf_llama3_1-8b.sh
```
Accuracy:
```
SCENARIO=Offline MODE=Accuracy    bash run_mlperf_llama3_1-8b.sh
SCENARIO=Server  MODE=Accuracy    bash run_mlperf_llama3_1-8b.sh
```

## Run Compliance Tests
Run this step inside the Docker container.  After the benchmark scenarios have been run and results exist in {LOG_DIR}/results, run this step to complete compliance runs. Compliance output will be found in '{LOG_DIR}/compliance'.
```
SCENARIO=Offline MODE=Compliance  bash run_mlperf_llama3_1-8b.sh
SCENARIO=Server  MODE=Compliance  bash run_mlperf_llama3_1-8b.sh
```

## Validate Submission Checker
Run this step inside the Docker container.  The following script will perform accuracy log truncation and run the submission checker on the contents of {LOG_DIR}. The source scripts are distributed as MLPerf Inference reference tools. Ensure the submission content has been populated before running.  The script output is transient and destroyed after running.  The original content of ${LOG_DIR} is not modified.
```
VENDOR=Intel SYSTEM=1-node-4x-BMG-Pro-B60-Dual bash scripts/prepare_submission.sh
