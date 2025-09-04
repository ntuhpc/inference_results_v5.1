# MLPerf Inference

This folder contains all of the code necessary to run:
 - MLPerf Inference Single-node "Offline"
 - MLPerf Inference Single-node "Server"
 - MLPerf Inference Multi-node "Offline"
 - MLPerf Inference Multi-node "Server"   

We use **FP8 quantized llama2-70b** model inference to generate all the results.  Please ensure that all software dependecies: ROCm 6.3.0, LLMBoost-0.5.2, Python 3.12 are installed. The benchmarking is run on **MI300X GPUs**. (8 GPUs MI300X GPUs for single node benchmarking, and four 8-MI300X nodes for multinode benchmarking).

The following steps outline the process of setting up a Docker environment and the details to reproduce our MLPerf V5.0 inference results.   

---

## 1. Docker Preparation

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

### Submission Environment

```bash
# Build the docker by
docker compose run --build llmboost-mlperf-5.0
```

---

## 2. Single Node Benchmarking:

**[AMD Code]** Please run the following code to reproduce our single node result on mlperf v5.0 inference.

```bash
cd /workspace/apps/mlperf/single_node/submission
bash llama2_70b.sh
```
The script will run both Offline and Server Scenarios for single node scenario.

**[LLMBoost Code]** Please run the following code:

```bash
cd /workspace/apps/mlperf
bash llama_70b_8x_mi300x.sh
```

**Or**, run with:

```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
SYSTEM_NAME=${SYSTEM_NAME:-"8xMI300X_2xEPYC_9534"} #CPU name: lscpu | grep name; count: lscpu | grep 'node(s)'

if [ -f 'audit.config' ]; then
   rm audit.config
fi

#Offline
# Performance
python3 mlperf.py \
    --model_name llama2-70b \
    --test_mode Offline \
    -tp 1 \
    -dp 8 \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/performance/run_1 \
    --user_conf conf/user_llama2-70b_8x_mi300x \
## Accuracy
python3 mlperf.py \
    --model_name llama2-70b \
    --test_mode Offline \
    -tp 1 \
    -dp 8 \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy \
    --user_conf conf/user_llama2-70b_8x_mi300x \
    --accuracy_test
bash tools/check_llama2_accuracy_scores.sh $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy/mlperf_log_accuracy.json
## Compliance TEST06
cp $TEST06_DIR/audit.config ./
python3 mlperf.py \
    --model_name llama2-70b \
    --test_mode Offline \
    -tp 1 \
    -dp 8 \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance -s Offline
rm audit.config

# Server
## Performance
python3 mlperf.py \
   --model_name llama2-70b \
   --test_mode Server \
   -tp 1 \
   -dp 8 \
   --drain_per_worker \
   --gpu_batch_size 48 \
   --batcher_threshold 0.2 \
   --load_balancing_mode batching \
   --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/performance/run_1 \
   --user_conf conf/user_llama2-70b_8x_mi300x

## Accuracy
python3 mlperf.py \
   --model_name llama2-70b \
   --test_mode Server \
   -tp 1 \
   -dp 8 \
   --drain_per_worker \
   --gpu_batch_size 48 \
   --batcher_threshold 0.2 \
   --load_balancing_mode batching \
   --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy \
   --user_conf conf/user_llama2-70b_8x_mi300x \
   --accuracy_test
bash tools/check_llama2_accuracy_scores.sh $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy/mlperf_log_accuracy.json
## Compliance TEST06
cp $TEST06_DIR/audit.config ./
python3 mlperf.py \
   --model_name llama2-70b \
   --test_mode Server \
   -tp 1 \
   -dp 8 \
   --drain_per_worker \
   --gpu_batch_size 48 \
   --batcher_threshold 0.2 \
   --load_balancing_mode batching \
   --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance -s Server
rm audit.config
```

---

## 3. Multi-Node (Default 4 Nodes):

Network is a collection 2 separate applications located in `apps/lon`, `server.py` and `client.py`. `server.py` is intended to be run on the AI-Box, `client.py` can either be run locally or on a remote system.

Please run the following code to reproduce our 4-node result on MLPerf v5.0 inference.

### 3.1. Offline Scenario

+ Please on each node, start the offline scenario server by:

```bash
cd /workspace/apps/mlperf
bash server_offline.sh
```

**Or**, run with:

```bash
cd /workspace/apps/mlperf
python3 server.py \
    -tp 1 \
    -dp 8 \
    -max_num_seqs 768 \
    --test_mode Offline
```

+ Then, on one of the node's docker or another machine, start the offline scenario benchmarking by (❗Please remember to specify the server IP address inside):

```bash
cd /workspace/apps/mlperf
bash client_offline.sh
```

**Or**, run with:

```bash
cd /workspace/apps/mlperf

SUBMISSION_DIR=/workspace/apps/mlperf/submission
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
SYSTEM_NAME=${SYSTEM_NAME:-"32xMI300X_2xEPYC_9534"} #CPU name: lscpu | grep name; count: lscpu | grep 'node(s)'
GPU_NAME=${GPU_NAME:-'mi300x'}
COMPANY=${COMPANY:-'MangoBoost'}

if [ -f 'audit.config' ]; then
   rm audit.config
fi

## Perf
python3 client.py \
    --test_mode Offline \
    --model_name llama2-70b \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/performance/run_1 \
    --user_conf conf/user_llama2-70b_32x_mi300x \
    --batched_queries 128 \
    --sut_server_addr "http://10.4.16.1:8000,http://10.4.16.2:8000,http://10.4.16.3:8000,http://10.4.16.4:8000"

## Accuracy
python3 client.py \
    --test_mode Offline \
    --model_name llama2-70b \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy  \
    --user_conf conf/user_32x_mi300.conf \
    --accuracy_test \
    --batched_queries 128 \
    --sut_server_addr="http://10.4.16.1:8000,http://10.4.16.2:8000,http://10.4.16.3:8000,http://10.4.16.4:8000"
bash tools/check_llama2_accuracy_scores.sh $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy/mlperf_log_accuracy.json

# Audit
cp $TEST06_DIR/audit.config ./
python3 client.py \
    --test_mode Offline \
    --model_name llama2-70b \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06 \
    --batched_queries 128 \
    --sut_server_addr="http://10.4.16.1:8000,http://10.4.16.2:8000,http://10.4.16.3:8000,http://10.4.16.4:8000"
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance -s Offline
rm audit.config
```

+ The results can be found in `./submission/results`.

#### 3.2. Server Scenario

+ Please on each node, start the server scenario server by:

```bash
cd /workspace/apps/mlperf
bash server_online.sh
```

**Or**, Run with:

```bash
cd /workspace/apps/mlperf
python3 server.py \
    -tp 1 \
    -dp 8 \
    -max_num_seqs 768 \
    --gpu_batch_size 48 \
    --batcher_threshold 0.2 \
    --load_balancing_mode batching \
    --test_mode Server
```

+ Then, on one of the node's docker or another machine, start the server scenario benchmarking by (❗Please remember to specify the server IP address inside):

```bash
cd /workspace/apps/mlperf
bash client_online.sh
```

**Or**, Run with:

````bash
cd /workspace/apps/mlperf

SUBMISSION_DIR=/workspace/apps/mlperf/submission
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
SYSTEM_NAME=${SYSTEM_NAME:-"32xMI300X_2xEPYC_9534"} #CPU name: lscpu | grep name; count: lscpu | grep 'node(s)'
GPU_NAME=${GPU_NAME:-'mi300x'}
COMPANY=${COMPANY:-'MangoBoost'}

if [ -f 'audit.config' ]; then
   rm audit.config
fi

## Perf
python3 client.py \
    --test_mode Server \
    --model_name llama2-70b \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/performance/run_1 \
    --user_conf conf/user_llama2-70b_32x_mi300x \
    --batched_queries 128 \
    --sut_server_addr "http://10.4.16.1:8000,http://10.4.16.2:8000,http://10.4.16.3:8000,http://10.4.16.4:8000"

# Accuracy
python3 client.py \
    --test_mode Server \
    --model_name llama2-70b \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy  \
    --user_conf conf/user_llama2-70b_32x_mi300x \
    --accuracy_test \
    --batched_queries 128 \
    --sut_server_addr="http://10.4.16.1:8000,http://10.4.16.2:8000,http://10.4.16.3:8000,http://10.4.16.4:8000"
bash tools/check_llama2_accuracy_scores.sh $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy/mlperf_log_accuracy.json

# Audit
cp $TEST06_DIR/audit.config ./
python3 client.py \
    --test_mode Server \
    --model_name llama2-70b \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06 \
    --batched_queries 128 \
    --sut_server_addr="http://10.4.16.1:8000,http://10.4.16.2:8000,http://10.4.16.3:8000,http://10.4.16.4:8000"
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance -s Server
rm audit.config
````

+ The results can be found in `./submission/results`.

----

## 4. Packing the Submission & Conducting Validation Checking

After the commands above, all the results are ready to be packaged. Please run the commands below to package the results and doing the submission checking for validation.

```bash
cd /workspace/apps/mlperf
bash submission_package.sh
```

+ If all the results are valid and all the submission directory structure is satisfied, it is ready to be zipped and submitted!