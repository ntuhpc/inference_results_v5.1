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
bash download_llama3_405b.sh
```

Inside the docker, quantize the model with

```bash
bash quantize_llama3_405b.sh
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

### Pruning the model

Llama3.1-405B has 126 layers, where each layer is architecturally the same. However, we have identified that some of the layers are less important to the output in comparison to the other layers. Using the calibration dataset, we measured the sum of the magnitude of the outputs of each of the transformer layers just before the finaly rmsnorm layer. 
$L_i = sum(abs(output_i))$
 where, $L_i$ is the importance of the $i^{th}$ layer and $output_i$ is the tensor of outputs of the transformer layers feed-forward network (just before the rmsnorm layer)

Based on $L_i$ of a small number of samples in the calibration dataset (we used 170 samples in our experiments), we drop several contiguous layers. This allows us to prune the model with higher number of layers while having minimum impact to the accuracy of the model output.


```
export $GIT_ROOT=<path_to_the_code_folder>
export $MODEL_PATH=<path_to_the_model>
python3 $GIT_ROOT/scripts/drop_layers.py --model $MODEL_PATH --initial 59 --final 84
```

The pruned model is saved in the folder  ```$MODEL_PATH/pruned_59_84/```

Note: what this script does in the above example is to prune layers of the model from layer 59 to layer 84 and save the remaining layers of the model into a folder named pruned_59_84. Note that we need to link the model path to this pruned model in vLLM configurations, i.e., in the offline_mi355x.yaml file in this case.

### Running the benchmark

Run the following commands inside the docker container

``` bash
## Performance
python /lab-mlperf-inference/code/llama3_1-405b_pruned/main.py \
   --config-path /lab-mlperf-inference/code/llama3_1-405b_pruned/harness_llm/models/llama3-1-405b/ \
   --config-name offline_mi355x \
   test_mode=performance \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama3_1-405b_pruned/user_mi355x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama3-1-405b/Offline/performance/run_1

## Accuracy
python /lab-mlperf-inference/code/llama3_1-405b_pruned/main.py \
   --config-path /lab-mlperf-inference/code/llama3_1-405b_pruned/harness_llm/models/llama3-1-405b/ \
   --config-name offline_mi355x \
   test_mode=accuracy \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama3_1-405b_pruned/user_mi355x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama3-1-405b/Offline/accuracy

### Evaluate accuracy
bash /lab-mlperf-inference/code/llama3_1-405b_pruned/scripts/check_llama3_accuracy_scores.sh \
   /lab-mlperf-inference/results/llama3-1-405b/Offline/accuracy/mlperf_log_accuracy.json
```
