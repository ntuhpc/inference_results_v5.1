### Running the benchmark

Run the following commands inside the docker container

``` bash
## Performance
python /lab-mlperf-inference/code/BENCHMARK_NAME/main.py \
   --config-path /lab-mlperf-inference/code/BENCHMARK_NAME/harness_llm/models/MODEL_NAME/ \
   --config-name scenario_GPU_NAME \
   --backend BACKEND \
   test_mode=performance \
   harness_config.device_count=GPU_COUNT \
   harness_config.user_conf_path=/lab-mlperf-inference/code/BENCHMARK_NAME/USER_CONF \
   harness_config.output_log_dir=/lab-mlperf-inference/results/MODEL_NAME/SCENARIO/performance/run_1

## Accuracy
python /lab-mlperf-inference/code/BENCHMARK_NAME/main.py \
   --config-path /lab-mlperf-inference/code/BENCHMARK_NAME/harness_llm/models/MODEL_NAME/ \
   --config-name scenario_GPU_NAME \
   --backend BACKEND \
   test_mode=accuracy \
   harness_config.device_count=GPU_COUNT \
   harness_config.user_conf_path=/lab-mlperf-inference/code/BENCHMARK_NAME/USER_CONF \
   harness_config.output_log_dir=/lab-mlperf-inference/results/MODEL_NAME/SCENARIO/accuracy

### Evaluate accuracy
bash /lab-mlperf-inference/code/BENCHMARK_NAME/scripts/ACCURACY_SCRIPT_NAME.sh \
   /lab-mlperf-inference/results/MODEL_NAME/SCENARIO/accuracy/mlperf_log_accuracy.json
```
