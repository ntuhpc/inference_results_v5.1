# MLPerf Inference 5.1

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

For vLLM V1 you need to enable the following flags:

```bash
vllm_env_config:
  VLLM_USE_V1: 1
  VLLM_V1_USE_PREFILL_DECODE_ATTENTION: 1
llm_config:
  num_scheduler_steps: 1
```

### Running partial runs

The following flags can be turned off if needed:
```bash
OFFLINE=1
SERVER=1
PERFORMANCE=1
ACCURACY=1
COMPLIANCE=1
PACKAGE=1
```

You need to define it before the command, e.g.:
```bash
OFFLINE=0 PACKAGE=0 bash /lab-mlperf-inference/submission/<benchmark>.sh
```

### Load balancing (offline benchmarks only)
Start the docker container for the benchmark by running the below command

```bash
bash setup/start_submission_env.sh $MLPERF_IMAGE_NAME
```

Run the below command in the container to measure performance per GPU

```bash
GPU_NAME="mi300x" RESULT_DIR="<output directory name>" bash /lab-mlperf-inference/code/scripts/measure_per_gpu_perf.sh BENCHMARK_NAME
```

Note: the results are usually consistent, so one the default one run is enough, but you can specify the number of runs by passing a second argument to the script, e.g. `bash /lab-mlperf-inference/code/scripts/measure_per_gpu_perf.sh BENCHMARK_NAME 3` will run the performance test 3 times.

The results will be stored in the following structure:
```
<output directory name>
    BENCHMARK_NAME_offline_performance_run_1_gpu_0
    ...
    BENCHMARK_NAME_offline_performance_run_1_gpu_7
    results.csv
    buckets.csv
```

Bucket sizes will also be printed out on the console.

If the terminal is cluttered after the script finished, re-initialize it with the following command:

```bash
reset
```

Set the derived bucket sizes in the `/lab-mlperf-inference/code/harness_llm/BENCHMARK_NAME/offline_<GPU_NAME>.yaml` file, e.g.:
```yaml
harness_config:
  ...
  sorting:
    strategy: buckets
    buckets: [
        # bucket size for GPUs in percentage separated by commas
        12.5,
        12.5,
        12.5,
        12.5,
        12.5,
        12.5,
        12.5,
        12.5
    ]
```

### Running experiments (Preview)
```bash
usage: submission.py [-h] --model {mixtral-8x7b,llama2-70b} {status,experiment,update_best,prepare,package} ...

MLPerf Inference Submission Tool

positional arguments:
  {status,experiment,update_best,prepare,package}
    status              Check best model state
    experiment          Run an experiment with the specified model and scenario
    update_best         Select the current best result for the specified model and scenario
    prepare             Run accuracy or compliance
    package             Package the current best results for submission.
                        It expects everything ready, check status before calling.
                        You have to set environment variables:
                        GPU_COUNT, GPU_NAME, CPU_COUNT, CPU_NAME, COMPANY.
                        The package will be created in the current directory.

options:
  -h, --help            show this help message and exit
  --model {mixtral-8x7b,llama2-70b}


usage: submission.py experiment [-h] --scenario {Offline,Server,Interactive} --model-conf MODEL_CONF --user-conf USER_CONF

options:
  -h, --help            show this help message and exit
  --scenario {Offline,Server,Interactive}
  --model-conf MODEL_CONF
                        Path to the model configuration YAML file
                        (e.g. code/harness_llm/models/llama2-70b/offline_mi325x.yaml).
  --user-conf USER_CONF
                        Path to the user configuration file
                        (e.g. code/user_mi325x.conf).


usage: submission.py update_best [-h] --scenario {Offline,Server,Interactive}

options:
  -h, --help            show this help message and exit
  --scenario {Offline,Server,Interactive}


usage: submission.py prepare [-h] --scenario {Offline,Server,Interactive} [--force] {accuracy,compliance}

positional arguments:
  {accuracy,compliance}

options:
  -h, --help            show this help message and exit
  --scenario {Offline,Server,Interactive}
  --force               Overwrite existing valid result

NOTE: package requires ENVs
See .submission_package_env for details
```
