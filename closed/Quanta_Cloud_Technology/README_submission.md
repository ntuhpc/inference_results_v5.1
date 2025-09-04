# MLPerf Inference Submission Helper

We provide a helper script (`submission.py`) that simplifies running MLPerf Inference benchmarks and preparing submission packages. This script is designed to streamline the workflow for both performance and compliance testing, and supports running multiple experiments with different configurations.

## Features

- Run inference benchmarks for supported models and scenarios.
- Prepare accuracy or compliance runs.
- Track and manage best run results.
- Package valid results for official submission.

## Supported Models

- `llama2-70b`
- `mixtral-8x7b`

## Supported Scenarios

- `Offline`
- `Server`
- `Interactive` (Only for llama2-70b)

---

## Example: Running a Benchmark Experiment

```bash
python submission.py --model llama2-70b experiment \
  --scenario Server \
  --model-conf /lab-mlperf-inference/code/harness_llm/models/llama2-70b/server_mi300x.yaml \
  --user-conf /lab-mlperf-inference/code/user_mi300x.conf

Usage

 General Command

python submission.py --model {llama2-70b,mixtral-8x7b} {experiment,prepare,status,package} [options]

>> --model is required and must be one of the supported models.

>> The second positional argument must be one of the following subcommands:

   - status
   
   - experiment

   - prepare

   - package

Subcommands


status
Check the current best-performing result for the model and scenario.


python submission.py --model llama2-70b status

experiment
Run a benchmark experiment for a specific model and scenario.


python submission.py --model llama2-70b experiment \
  --scenario {Offline,Server,Interactive} \
  --model-conf <path_to_model_conf.yaml> \
  --user-conf <path_to_user_conf.conf>

prepare
Prepare a run for accuracy or compliance validation.


python submission.py --model llama2-70b prepare --scenario Server 


package
Package the best result for submission. This requires specific environment variables to be set.


python submission.py --model llama2-70b package
Note: The package step requires the following environment variables:

export GPU_COUNT="8"
export GPU_NAME="mi300x"
export CPU_COUNT="2"
export CPU_NAME="EPYC_9575F"
export COMPANY="AMD"

You can either export them in your shell or define them in a .submission_package_env file and then source.

