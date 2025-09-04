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
bash download_mixtral_8x7b.sh
```

Inside the docker, quantize the model with

```bash
bash quantize_mixtral_8x7b.sh
```

Exit the docker image, because a different image is needed for inference

## Inference

### Runtime tunables

To boost the machine's performance further, execute the following script before any performance test (should be set once after a reboot):

```bash
bash setup/runtime_tunables.sh
```

### Docker

```bash
export MLPERF_IMAGE_NAME=rocm/mlperf-inference:submission_5.1-mixtral_8x7b
```

Build the docker image for the benchmark by running the below command

```bash
bash setup/build_submission_mixtral_8x7b.sh $MLPERF_IMAGE_NAME
```

Start the docker container for the benchmark by running the below command

```bash
bash setup/start_submission_env.sh $MLPERF_IMAGE_NAME
```

### Running the benchmark and submission packaging

We provide helper scripts to run the benchmark and create submission packages for mixtral-8x7b ([mixtral_8x7b.sh](./submission/mixtral_8x7b.sh)).

The package will be generated in the `submission/inference_results_5.1` folder. This folder will contain all the results and information to recreate the results.

Run the below command in the container

```bash
# mixtral_8x7b GPU_NAME can be mi300x/mi325x.
# Set CPU_NAME based on your hardware you use. You can use `lscpu | grep name`.
# GPU_COUNT can be 1 or 8. 1 is only applicable for llama2, llama2-interactive and mixtral-8x7b.
# RESULTS can be set to the ouput for the results. By default it is set to the results directory of the CWD (/lab-mlperf-inference/submission/results).
# ENABLE_POWER_SETUP is used to set GPU frequency and power state to a predetermined value for best performace. By default it is set to `1` set it to `0` to disable it.
COMPANY="<your company name>" CPU_NAME="EPYC_9554" GPU_NAME="mi300x" GPU_COUNT=8 RESULTS="<output directory name>" ENABLE_POWER_SETUP=1 bash /lab-mlperf-inference/submission/mixtral_8x7b.sh
```

To run the packaging script only, run the below command in the container
```bash
# The parameters are the same as above described
COMPANY="<your company name>" CPU_NAME="EPYC_9554" GPU_NAME="mi300x" GPU_COUNT=8 RESULTS="<output directory name>" bash /lab-mlperf-inference/submission/package_submission.sh
```
