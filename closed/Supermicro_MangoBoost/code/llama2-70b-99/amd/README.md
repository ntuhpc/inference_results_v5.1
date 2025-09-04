# MLPerf Inference 5.0

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

### Docker

Build the docker image for the benchmark by running the below command

```bash
bash setup/build_submission_env.sh
```

### LLMBoost

LLMBoost can be installed from the given wheel file (`llmboost-0.5.2+py3.12-py3-none-any.whl`).
```
# Attach to the docker container
bash setup/start_submission_env.sh

# Make sure that the Python version is 3.12.x
# python3 --version
Python 3.12.8

$ python3 -m pip install llmboost-0.5.2-py3-none.any.whl

$ which llmboost
/usr/local/bin/llmboost
```

## Inference

### Docker

Start the docker container for the benchmark by running the below command 

TODO - Peiran

### Running the benchmark and submission packaging
     
We provide helper scripts to run the benchmark and create submission packages for llama2-70b ([llama2_70b.sh](./submission/llama2_70b.sh)).

The package will be generated in the `submission/inference_results_5.0` folder. This folder will contain all the results and information to recreate the results.

Run the below command in the container

```bash
# llama2_70b GPU_NAME can be mi300x/mi325x set SYSTEM_NAME based on your hardware you use
COMPANY="<your company name>" SYSTEM_NAME="8xMI300X_2xEPYC_9554" GPU_NAME="mi300x" bash /lab-mlperf-inference/submission/llama2_70b.sh
```

## Tweaking and troubleshooting the benchmark

> [!IMPORTANT]
> DANGER ZONE
> If you are not sure how to modify the benchmark configuration, get in touch with your amd contact.

### Modifying the qps for the server scenario

The target_qps, query per second, parameter for the server scenario controls the load that loadgen sends us. If you see in the performance logs that your ttft, time to first token, and tpot, time per output token, values are significantly lower than the benchmark requirements, then you can try to increase the qps parameter telling loadgen to send samples faster to the benchmark that will potentially result in a higher throughput. You can find this config in [user_mi300x.conf](./code/user_mi300x.conf) and [user_mi325x.conf](./code/user_mi325x.conf).

```yml
# Example value
llama2-70b.Server.target_qps = 79
```

### Latency requirements fail for the server scenario

If you see the below lines in your performance results, it means that your system cannot process the samples sent by loadgen within the required time, your througput cannot handle the load.

```
Result is : INVALID
  Performance constraints satisfied : No
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: No
```

First, try to use a lower target_qps parameter for the server scenario than the currently configured one. See the section [Modifying the qps for the server scenario](###-Modifying-the-qps-for-the-server-scenario) for the details of how to do this.

### Longer runs for the offline scenario

During an offline scenario run, the gpus are saturated for most of the time except at the end of the run when their load is decreasing gradually. The cause of this is that different samples will generate responses of varying length. As we are closing to the end of the run fewer and fewer prompts need further processing that results in underutilized gpus that might decrease our overall throughput by a few percent. If we instruct loadgen to send more samples, the whole run will be longer and the underutilized end interval becomes less and less relevant. We can achieve this by increasing the min_duration for the offline scenario. You can find this config in [user_mi300x.conf](./code/user_mi300x.conf) and [user_mi325x.conf](./code/user_mi325x.conf).

```yml
# Example value for 20 minutes run
llama2-70b.Offline.min_duration = 1200000
```

Note: You need to closely match the samples per second result. If you managed to get a better result, you might need to increase the target_qps value in the conf to be valid.

### The harness cannot find the model or the dataset

The harness will look for the model and the dataset in preconfigured places. Our helper scripts for starting the docker containers, downloading the model and dataset use these paths and put all the data in the configured locations. You can check which directory is mounted for the model and dataset in the [start_submission_env.sh](./setup/start_submission_env.sh)

```bash
export LAB_MODEL="${LAB_MODEL:-/data/inference/model/}"
export LAB_DATASET="${LAB_DATASET:-/data/inference/data/}"
```

These directories are mounted under the _/model_ and _/data_ paths in the container respectively. The harness will look for the model and the dataset under these paths by the following subpaths that you check in harness configuration files under _code/harness_llm/models/llama2-70b_

```yml
...
# Example configuration for the model path
llm_config:
  model: /model/llama2-70b-chat-hf/fp8_quantized/
...
# Example configuration for the dataset path
harness_config:
  dataset_path: /data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
...
```
If you encounter any issues with model and dataset loading, check the above docker environment and harness configurations and make sure that you placed your data in the right places.

### The benchmark run seems to be hanging or stuck

Even the performance benchmark run can take a very long time, accuary runs take even more. This can be hours. Logging is minimized or turned off for some intervals during the run to avoid hurting the performance. If you are not sure that everything is OK with the benchmark run, we recommended to issue the below command on the system.

```bash
watch -n 0.1 rocm-smi
```

If you see that the gpus are working, then everything is fine. If you see that only a portion of the gpus are working, that most likely means that you run is close to the end and several gpus already processed their assigned load.

### Tweaking vllm

> [!IMPORTANT]
> EXPERTS ONLY

The harness uses [vllm](https://github.com/vllm-project/vllm) as our llm backend. If you are familiar with it and you would like to tweak the parameters passed to the vllm engine, you can do that through the harness configuration files under _code/harness_llm/models/llama2-70b_

```bash
# Example configuration passed to vllm in the harness
llm_config:
  model: /model/llama2-70b-chat-hf/fp8_quantized/
  tensor_parallel_size: 1
  num_scheduler_steps: 9
  quantization: fp8
  max_model_len: 2048
  swap_space: 0
  gpu_memory_utilization: 0.96
  max_seq_len_to_capture: 2048
  enforce_eager: True
  disable_custom_all_reduce: True
  max_num_batched_tokens: 26624
  max_num_seqs: 1024
  enable_chunked_prefill: False
  block_size: 16
  enable_prefix_caching: False
```

Everything under the _llm_config_ node is passed as is to the vllm engine. The paramater names are the names that you find in the [vllm engine params](https://github.com/vllm-project/vllm/blob/83481ceb499b105b11e7ae1507a409c6f7711bfa/vllm/engine/arg_utils.py#L89). You edit remove or add parameters that are not used currently by the harness. 

Please be advised that not every vllm feature is compliant with the mlperf inference rules! For example vllm's prefix caching is not allowed to use. Before any change to the configuration, consult the [documentation](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) or your amd contact if you are not sure that your change is compliant.
