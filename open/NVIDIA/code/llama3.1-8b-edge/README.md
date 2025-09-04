# Thor Docker Workflow for MLPerf Inference

This document outlines the complete workflow for building and running MLPerf inference using the Thor Docker container with drive-llm integration.

## Prerequisites

- Access to NGC Thor base image is currently pending GA release but we've beta software to our partners. The GA release will come in the middle of Aug
- Thor hardware platform with CUDA 13.0 driver
- Required data directories and model files

## Step 1: Navigate to Project Directory

```bash
cd /mlperf-inference/closed/NVIDIA
# Create local fast storage directory for reading/writing engines
mkdir -p /home/engines
```

## Step 2: Prepare Required Data

Before launching the container, ensure you have downloaded the required datasets and models:

- **Dataset**: Follow the [MLPerf Inference official guide](https://github.com/mlcommons/inference/tree/small_llm_reference/language/llama3.1-8b#5000-samples-edge) to download the 5000 samples edge dataset
- **Model and Tokenizer**: Download from [Hugging Face Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)

## Step 3: Launch Thor Docker Container

Run the Thor Docker container with proper volume mounts:

```bash
make launch_thor_docker THOR_BASE_IMAGE_REGISTRY=nvcr.io/nvidia/mlperf/mlperf-inference
```

The launch command automatically mounts the following directories:
- `/home/engines:/home/engines` - Model weights and engines storage
- User home directory for SSH access
- Various device access points for Thor hardware

### Alternative Manual Launch (Optional)

If you need to customize mount paths, refer to the `launch_thor_docker` target in `Makefile.docker` and modify the volume mounts accordingly.

## Step 4: Container Setup

Once inside the container, navigate to the llama3.1-8b directory:

```bash
cd code/llama3.1-8b-edge
# follow the mlperf inference offical guide to download dataset https://github.com/mlcommons/inference/tree/small_llm_reference/language/llama3.1-8b#5000-samples-edge, 
#as well as tokenizer and model from https://huggingface.co/meta-llama/Llama-3.1-8B

```

Ensure that the required datasets, tokenizer, and model files are properly accessible within the container environment.

## Step 5: Build W4A16 Engine

Generate the optimized W4A16 engine using the prepackaged ONNX file:

```bash
bash W4A16_engine_build.sh
```

This script builds the TensorRT engine with W4A16 precision optimizations specifically for the Thor platform.

## Step 6: Run Performance and Accuracy Tests

Execute the test suite to validate performance and accuracy:

```bash
# Run performance benchmark
bash run_thor_singlestream_perf.sh

# Run accuracy evaluation
bash run_thor_accuracy.sh
```

## Step 7: Accuracy Evaluation Process

The accuracy evaluation follows a multi-step process:

### 1. Generate Accuracy Log
MLPerf LoadGen automatically dumps the accuracy data to `output/mlperf_log_accuracy.json`. This file stores the output tokens for each sample in a numpy array format.

### 2. Run Evaluation Script
Once the accuracy file is generated, use `evaluation.py` to compare against the original dataset accuracy using ROUGE score as the metric:

```bash
python evaluation.py \
    --mlperf-accuracy-file output/mlperf_log_accuracy.json \
    --dataset-file=<path_to_cnn_daily_mail_5000_samples.json> \
    --dtype int32 \
    --model-name=<path_to_model_with_tokenizers>
```

### 3. Evaluation Results
The evaluation script will:
- Load the generated `mlperf_log_accuracy.json` file
- Compare the model outputs against the ground truth from the original dataset
- Calculate ROUGE scores to measure accuracy
- Provide detailed accuracy metrics for the MLPerf inference run

## Troubleshooting

- Ensure all required data files are properly mounted and accessible
- Verify CUDA 13.0 driver compatibility
- Check that the Thor hardware platform is properly configured
- Confirm all volume mounts are correctly set up in the Docker container
