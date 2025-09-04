# Supermicro_MangoBoost MLPerf Inference V5.1

This folder contains the detailed instructions to reproduce our following MLPerf submissions:
 1. **Homogeneous Multi-node** (16X MI325X) *Offline* and *Server* Scenerios
 2. **Heterogeneous Multi-node** (16X MI325X + 8X MI300X) *Offline* and *Server* Scenerios

The following steps outline the processes of setting up a Docker environment and the details to reproduce Supermicro and MangoBoost joint submission of MLPerf V5.1 inference results.   

---

## 1. Preparation Before Benchmarking

### 1.1 Model and Dataset Preparation

Please download the model and dataset according to AMD's guideline [link](https://rocm.blogs.amd.com/artificial-intelligence/reproducing-amd-mlperf-inference-submission/README.html). 

### 1.2 LLMBoost Docker Preparation

Our **feature-restricted** docker is available in DockerHub, please pull the docker using the following command:
```bash
docker pull llmboost/mb-llmboost:mlperf-5.1
```
> ***Important Note***: This is a **feature-restricted** docker of the software stack [Mango LLMBoost](https://www.mangoboost.io/products/software/mango-llmboost-tm), which only enables the functionality and performance on MLPerf llama2-70B inference benchmarking. To unlock a full version of LLMBoost, please contact MangoBoost support at [contact@mangoboost.io](contact@mangoboost.io)!

Then, please use this command to run the docker container:

```bash
docker run -it --rm \
    --network host \
    --group-add video \
    --ipc host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/dri:/dev/dri \
    --device=/dev/kfd:/dev/kfd \
    -v <path to quantized llama2-70b models>:/models/amd2025_model/model/llama2-70b-chat-hf/quantized \
    -v <path to the processed llama2-70b dataset>:/models/amd2025_model/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl \
    llmboost/mb-llmboost:mlperf-5.1
```

***From here on, we assume all the belowing commands run within the LLMBoost docker container.***

---

## 2. Homogeneous Multi-Node (16X MI325X):

The commands to run benchmarking on multi-node is basically the same as on single-node, except for need to start the services on several nodes. Please follow the instructions below to reproduce the result on MLPerf v5.1 inference.

### 2.1 Homogeneous Multi-Node *Offline* Scenario Benchmarking

Please on each MI325X node, start the LLMBoost service by running the command:
```bash
cd /workspace/apps/mlperf
python3 server.py --test_mode Offline --model_path "/models/amd2025_model/model/llama2-70b-chat-hf/quantized" --accelerator_name mi325x
```

Then, wait until all the nodes finish the intialization and listening on the port `0.0.0.0:8000` and `0.0.0.0:8001`.

### 2.1.1 Homogeneous Multi-Node *Offline* Performance Run
With the LLMBoost service on, you can start a separate terminal on any machines (can be one of the machines that you started the LLMBoost services). Then, please run the following commands to start the performance run:
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_2xEPYC_9575F"}
cd /workspace/apps/mlperf
python3 client.py \
    --test_mode Offline \
    --user_conf conf/user_llama2-70b_16x_mi325x.conf \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/performance/run_1 \
    --sut_server_addr "http://<mi325x-n1>,http://<mi325x-n2>"
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

### 2.1.2 Homogeneous Multi-Node *Offline* Accuracy Test

Please use the following command to run the accuracy test on multi-node.
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_2xEPYC_9575F"}
cd /workspace/apps/mlperf
python3 client.py \
    --test_mode Offline \
    --user_conf conf/user_llama2-70b_16x_mi325x.conf \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy  \
    --accuracy_test \
    --sut_server_addr "http://<mi325x-n1>,http://<mi325x-n2>"
bash tools/check_llama2_accuracy_scores.sh $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy/mlperf_log_accuracy.json
cat $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy/accuracy.txt
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

This command will output the rouge score in the end, and please make sure the score is above the constraint.

### 2.1.3 Homogeneous Multi-Node *Offline* Audit Test

Please use the following command to run the audit test on multi-node.
```bash
 SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_2xEPYC_9575F"}
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
cp $TEST06_DIR/audit.config ./
python3 client.py \
    --test_mode Offline \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06  \
    --sut_server_addr "http://<mi325x-n1>,http://<mi325x-n2>"
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance -s Offline
rm audit.config
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

The audit test only sends 100 samples to LLMBoost services, which will be really quick. You can check whether it pass or fail according to the output file `$SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06/verify_accuracy.txt`. 

### 2.2 Homogeneous Multi-Node *Server* Scenario Benchmarking

Please on each MI325X node, start the LLMBoost service by running the command:
```bash
cd /workspace/apps/mlperf
python3 server.py --test_mode Server --model_path "/models/amd2025_model/model/llama2-70b-chat-hf/quantized" --accelerator_name mi325x
```

Then, wait until both of the nodes finish the intialization and listening on the port `0.0.0.0:8000` and `0.0.0.0:8001`.

### 2.2.1 Homogeneous Multi-Node *Server* Performance Run
With the LLMBoost service on, you can start a separate terminal on any one of the machines that you started the LLMBoost service. Then, please run the following commands to start the performance run:
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_2xEPYC_9575F"}
cd /workspace/apps/mlperf
python3 client.py \
    --test_mode Server \
    --user_conf conf/user_llama2-70b_16x_mi325x.conf \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/performance/run_1 \
    --sut_server_addr "http://<mi325x-n1>,http://<mi325x-n2>" 
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

### 2.2.2 Homogeneous Multi-Node *Server* Accuracy Test

Please use the following command to run the accuracy test on multi-node.
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_2xEPYC_9575F"}
cd /workspace/apps/mlperf
python3 client.py \
    --test_mode Server \
    --user_conf conf/user_llama2-70b_16x_mi325x.conf \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy  \
    --accuracy_test \
    --sut_server_addr "http://<mi325x-n1>,http://<mi325x-n2>"
bash tools/check_llama2_accuracy_scores.sh $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy/mlperf_log_accuracy.json
cat $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy/accuracy.txt
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

This command will output the rouge score in the end, and please make sure the score is above the constraint.

### 2.2.3 Homogeneous Multi-Node *Server* Audit Test

Please use the following command to run the audit test on multi-node.
```bash
 SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_2xEPYC_9575F"}
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
cp $TEST06_DIR/audit.config ./
python3 client.py \
    --test_mode Server \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06  \
    --sut_server_addr "http://<mi325x-n1>,http://<mi325x-n2>"
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance -s Server
rm audit.config
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

The audit test only sends 100 samples to LLMBoost services, which will be really quick. You can check whether it pass or fail according to the output file `$SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06/verify_accuracy.txt`. 

---

## 3. Heterogeneous Multi-Node (16X MI325X + 8X MI300X):

The commands to run benchmarking on multi-node is basically the same as on single-node, except for need to start the services on several nodes. Please follow the instructions below to reproduce the result on MLPerf v5.1 inference.

### 3.1 Heterogeneous Multi-Node *Offline* Scenario Benchmarking

Please on each MI325X node, start the LLMBoost service by running the command:
```bash
cd /workspace/apps/mlperf
python3 server.py --test_mode Offline --model_path "/models/amd2025_model/model/llama2-70b-chat-hf/quantized" --accelerator_name mi325x
```

Please on the MI300X node, start the LLMBoost service by running the command:
```bash
cd /workspace/apps/mlperf
python3 server.py --test_mode Offline --model_path "/models/amd2025_model/model/llama2-70b-chat-hf/quantized" --accelerator_name mi300x
```

Then, wait until all the nodes finish the intialization and listening on the port `0.0.0.0:8000` and `0.0.0.0:8001`.

### 3.1.1 Heterogeneous Multi-Node *Offline* Performance Run
With the LLMBoost service on, you can start a separate terminal on any machines (can be one of the machines that you started the LLMBoost services). Then, please run the following commands to start the performance run:
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_8xMI300X_2xEPYC_9575F"}
cd /workspace/apps/mlperf
python3 client.py \
    --test_mode Offline \
    --user_conf conf/user_llama2-70b_16x_mi325x_8x_mi300x.conf \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/performance/run_1 \
    --sut_server_addr "http://<mi300x-n1>,http://<mi325x-n1>,http://<mi325x-n2>" \
    --scheduler weighted_random \
    --scheduler_weights "5,6,6"
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

### 3.1.2 Heterogeneous Multi-Node *Offline* Accuracy Test

Please use the following command to run the accuracy test on multi-node.
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_8xMI300X_2xEPYC_9575F"}
cd /workspace/apps/mlperf
python3 client.py \
    --test_mode Offline \
    --user_conf conf/user_llama2-70b_16x_mi325x_8x_mi300x.conf \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy  \
    --accuracy_test \
    --sut_server_addr "http://<mi300x-n1>,http://<mi325x-n1>,http://<mi325x-n2>" \
    --scheduler weighted_random \
    --scheduler_weights "5,6,6"
bash tools/check_llama2_accuracy_scores.sh $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy/mlperf_log_accuracy.json
cat $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy/accuracy.txt
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

This command will output the rouge score in the end, and please make sure the score is above the constraint.

### 3.1.3 Heterogeneous Multi-Node *Offline* Audit Test

Please use the following command to run the audit test on multi-node.
```bash
 SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_8xMI300X_2xEPYC_9575F"}
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
cp $TEST06_DIR/audit.config ./
python3 client.py \
    --test_mode Offline \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06  \
    --sut_server_addr "http://<mi300x-n1>,http://<mi325x-n1>,http://<mi325x-n2>" \
    --scheduler weighted_random \
    --scheduler_weights "5,6,6"
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance -s Offline
rm audit.config
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

The audit test only sends 100 samples to LLMBoost services, which will be really quick. You can check whether it pass or fail according to the output file `$SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06/verify_accuracy.txt`. 

### 3.2 Heterogeneous Multi-Node *Server* Scenario Benchmarking

Please on each MI325X node, start the LLMBoost service by running the command:
```bash
cd /workspace/apps/mlperf
python3 server.py --test_mode Server --model_path "/models/amd2025_model/model/llama2-70b-chat-hf/quantized" --accelerator_name mi325x
```

Then, wait until both of the nodes finish the intialization and listening on the port `0.0.0.0:8000` and `0.0.0.0:8001`.

### 3.2.1 Heterogeneous Multi-Node *Server* Performance Run
With the LLMBoost service on, you can start a separate terminal on any one of the machines that you started the LLMBoost service. Then, please run the following commands to start the performance run:
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_8xMI300X_2xEPYC_9575F"}
cd /workspace/apps/mlperf
python3 client.py \
    --test_mode Server \
    --user_conf conf/user_llama2-70b_16x_mi325x_8x_mi300x.conf \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/performance/run_1 \
    --sut_server_addr "http://<mi300x-n1>,http://<mi325x-n1>,http://<mi325x-n2>" \
    --scheduler weighted_random \
    --scheduler_weights "5,6,6"
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

### 3.2.2 Heterogeneous Multi-Node *Server* Accuracy Test

Please use the following command to run the accuracy test on multi-node.
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_8xMI300X_2xEPYC_9575F"}
cd /workspace/apps/mlperf
python3 client.py \
    --test_mode Server \
    --user_conf conf/user_llama2-70b_16x_mi325x_8x_mi300x.conf \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy  \
    --accuracy_test \
    --sut_server_addr "http://<mi300x-n1>,http://<mi325x-n1>,http://<mi325x-n2>" \
    --scheduler weighted_random \
    --scheduler_weights "5,6,6"
bash tools/check_llama2_accuracy_scores.sh $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy/mlperf_log_accuracy.json
cat $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy/accuracy.txt
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

This command will output the rouge score in the end, and please make sure the score is above the constraint.

### 3.2.3 Heterogeneous Multi-Node *Server* Audit Test

Please use the following command to run the audit test on multi-node.
```bash
 SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"16xMI325X_8xMI300X_2xEPYC_9575F"}
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
cp $TEST06_DIR/audit.config ./
python3 client.py \
    --test_mode Server \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06  \
    --sut_server_addr "http://<mi300x-n1>,http://<mi325x-n1>,http://<mi325x-n2>" \
    --scheduler weighted_random \
    --scheduler_weights "5,6,6"
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance -s Server
rm audit.config
```
> Note: Please specify the --sut_server_addr according to the IP address of your nodes.

The audit test only sends 100 samples to LLMBoost services, which will be really quick. You can check whether it pass or fail according to the output file `$SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06/verify_accuracy.txt`. 

---

## 4. Packing the Submission & Conducting Validation Checking

After the commands above, all the results are ready to be packaged. Please run the commands below to package the results and doing the submission checking for validation.

```bash
cd /workspace/apps/mlperf
bash submission_package.sh
```