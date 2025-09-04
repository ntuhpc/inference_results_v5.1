# ResNet-50 Inference on iGPU using OpenVINO

## LEGAL DISCLAIMER
To the extent that any data, datasets, or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality. By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license. 

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets, or models. 

## Add the OpenVINO Python Backend to the MLPerf Inference Reference Repository
Copy the files within `python/` to `inference/vision/classification_and_detection/python/`. This will add the OpenVINO backend and update `main.py` with the OpenVINO profile.

```
mkdir ~/mlperf-ov && cd ~/mlperf-ov
git clone https://github.com/mlcommons/inference.git --recursive
cd ./inference/vision/classification_and_detection
cp ${PATH-TO-SHARED-PYTHON-FILES}/backend_openvino.py .
cp ${PATH-TO-SHARED-PYTHON-FILES}/main.py .
```

## Set up MLPerf OpenVINO Environment
Run the setup_env.sh script to download the MLPerf models, sample datasets, and required packages.
```
cd ./vision/classification_and_detection/
./install_prerequisites.sh
./setup_env.sh
source mlperf_env/bin/activate
```

## Sample Command Script
Use the sample script to run the ResNet-50 SingleStream scenario on the iGPU.
```
./sample_cmdline_resnet50.sh
```

## Run Benchmark
Run this step after sourcing the mlperf_env environment and after setting the `OPENVINO_DEVICE` environment variable.  This implementation currently only works for the SingleStream Scenario.

Before running any workload, verify the `OPENVINO_DEVICE` environment variable is set to either CPU, GPU, or NPU. Please refer to the example below for iGPU.
```
export OPENVINO_DEVICE=GPU

# Performance Testing
python3 ./python/main.py --profile resnet50-openvino --model ${MODEL-PATH}/resnet50_int8.xml --dataset-path ${DATASET-PATH}/imagenet/ --output results/ResNet50_INT8_iGPU_SingleStream/performance --scenario SingleStream --max-batchsize=1 --samples-per-query=1

# Accuracy Testing
python3 ./python/main.py --profile resnet50-openvino --model ${MODEL-PATH}/resnet50_int8.xml --dataset-path ${DATASET-PATH}/imagenet/ --output results/ResNet50_INT8_iGPU_SingleStream/accuracy --scenario SingleStream --max-batchsize=1 --samples-per-query=1 --accuracy
```

## Run Compliance Tests
Run this step after sourcing the mlperf_env environment and after setting the `OPENVINO_DEVICE` environment variable.  This implementation currently only works for the SingleStream Scenario.

### TEST01

```
cp ../../compliance/nvidia/TEST01/resnet50/audit.config .
python3 ./python/main.py --profile resnet50-openvino --model ${MODEL-PATH}/resnet50_int8.xml --dataset-path ${DATASET-PATH}/imagenet/ --output audit/TEST01/ResNet50_INT8_iGPU_SingleStream/ --scenario SingleStream --max-batchsize=1 --samples-per-query=1

python3 ../../compliance/nvidia/TEST01/run_verification.py -r=./results/ResNet50_INT8_iGPU_SingleStream -c=./audit/TEST01/ResNet50_INT8_iGPU_SingleStream/ -o=./compliance
```

### TEST04

```
cp ../../compliance/nvidia/TEST04/audit.config .
python3 ./python/main.py --profile resnet50-openvino --model ${MODEL-PATH}/resnet50_int8.xml --dataset-path ${DATASET-PATH}/imagenet/ --output audit/TEST04/ResNet50_INT8_iGPU_SingleStream/ --scenario SingleStream --max-batchsize=1 --samples-per-query=1

python3 ../../compliance/nvidia/TEST01/run_verification.py -r=./results/ResNet50_INT8_iGPU_SingleStream -c=./audit/TEST04/ResNet50_INT8_iGPU_SingleStream/ -o=./compliance
```
