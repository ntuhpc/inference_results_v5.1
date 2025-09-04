# ResNet-50 — Offline — MLPerf Inference v5.1 — Apple M1 Pro


This README documents the exact environment, dataset/model preparation, and MLPerf pipeline command used to reproduce the **Offline** scenario results for the Edge/Open division.

## Step 1: Environment Setup

We use **conda** to manage dependencies and ONNX Runtime with the CoreML execution provider for Apple M1 GPU/ANE acceleration.


```bash
# Create and activate Python 3.11 environment
conda create -n <your_env_name> python=3.11 -y
conda activate <your_env_name>
python -m pip install -U pip

# Install MLCommons automation tools and ONNX Runtime
pip install mlc-scripts mlcflow onnxruntime==1.22.1

# Enable CoreML EP (GPU + ANE)
export ORT_ENABLE_COREML=1
export ORT_COREML_USE_GPU_ONLY=0  
export THREADS=8     

```

Check available execution providers:

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
# Expected: ['CoreMLExecutionProvider', 'CPUExecutionProvider']

```

## Step 2: Dataset Preparation

We use the preprocessed ImageNet-1k validation set. Before launching the container, ensure you have downloaded the required datasets downloaded. 

- **Dataset**: Follow the [MLPerf Inference official guide](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) to download the imagenet2012 (validation) dataset

```bash
# Clone MLPerf Inference repository
git clone https://github.com/mlcommons/inference.git
cd inference

# Set ImageNet root path 
export IMAGENET_ROOT=</absolute/path/to/imagenet>

# Download and preprocess validation set
mlcr get,dataset,imagenet,validation,_full -j --outdirname="${IMAGENET_ROOT}/preprocessed"

```

## Step 3: Model Preparation


```bash
mlcr get,ml-model,resnet50,image-classification,_onnx -j --outdirname=vision/classification_and_detection/pretrained_models/onnx

```





## Step 4: Running the Offline Scenario


```bash
mlcr run-mlperf,inference,_full,_r5.1-dev \
     --model=resnet50 --implementation=reference \
     --framework=onnxruntime --category=edge \
     --scenario=Offline --device=cpu --threads=${THREADS} \
     --dataset-path="${IMAGENET_ROOT}/preprocessed" \
     --execution_mode=valid

# Even with --device=cpu, CoreML EP will internally use Apple GPU/ANE

```

Ensure that the required datasets and model files are properly accessible within the container environment.

