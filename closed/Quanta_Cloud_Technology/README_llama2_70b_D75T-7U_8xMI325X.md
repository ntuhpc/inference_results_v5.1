# MLPerf Inference 5.1

## Setup

### Model and Dataset

Build the docker image for the benchmark by running the below command

```bash
bash setup/build_model_and_dataset_env.sh
```

Start the docker container for the benchmark by running the below command

```bash
bash setup/start_model_and_dataset_env.sh
```

Inside the docker, download the model with

```bash
# Generate an access token on huggingface and set it here
HUGGINGFACE_ACCESS_TOKEN="<your HF token goes here>" python download_model.py
```

Inside the docker, download the dataset with

```bash
bash download_llama2_70b.sh
```

Inside the docker, quantize the model with

```bash
bash quantize_llama2_70b.sh
```

Exit the docker image, because a different image is needed for inference

## Inference

### Runtime tunables

To boost the machine's performance further, execute the following script before any performance test (should be set once after a   reboot):

```bash
bash setup/runtime_tunables.sh
```

### Docker

```bash
export MLPERF_IMAGE_NAME=rocm/mlperf-inference:submission_5.1-llama2_70b
```

Build the docker image for the benchmark by running the below command

```bash
bash setup/build_submission_llama2_70b.sh $MLPERF_IMAGE_NAME
```

Start the docker container for the benchmark by running the below command

```bash
bash setup/start_submission_env.sh $MLPERF_IMAGE_NAME
```

##After this please refer to README_submission.md


