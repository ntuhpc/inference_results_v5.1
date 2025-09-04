# MLPerf Inference 5.1

## Setup

Follow the instruction in code/llama3_1-405b_finetuned/mlperf_finetune/README_mlperf.md

### Docker

```bash
export MLPERF_IMAGE_NAME=rocm/mlperf-inference:submission_5.1-llama3_405b
```

Build the docker image for the benchmark by running the below command

```bash
bash setup/build_submission_llama3_405b.sh $MLPERF_IMAGE_NAME
```

Start the docker container for the benchmark by running the below command

```bash
bash setup/start_submission_env.sh $MLPERF_IMAGE_NAME
```

### Running the benchmark

Run the following commands inside the docker container

``` bash
## Performance
python /lab-mlperf-inference/code/llama3_1-405b_finetuned/main.py \
   --config-path /lab-mlperf-inference/code/llama3_1-405b_finetuned/harness_llm/models/llama3-1-405b/ \
   --config-name offline_mi355x \
   test_mode=performance \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama3_1-405b_finetuned/user_mi355x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama3-1-405b/Offline/performance/run_1

## Accuracy
python /lab-mlperf-inference/code/llama3_1-405b_finetuned/main.py \
   --config-path /lab-mlperf-inference/code/llama3_1-405b_finetuned/harness_llm/models/llama3-1-405b/ \
   --config-name offline_mi355x \
   test_mode=accuracy \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama3_1-405b_finetuned/user_mi355x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama3-1-405b/Offline/accuracy

### Evaluate accuracy
bash /lab-mlperf-inference/code/llama3_1-405b_finetuned/scripts/check_llama3_accuracy_scores.sh \
   /lab-mlperf-inference/results/llama3-1-405b/Offline/accuracy/mlperf_log_accuracy.json
```
