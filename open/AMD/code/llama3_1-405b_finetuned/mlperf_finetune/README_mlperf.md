# MLPerf Inference 5.1

## Setup

### Docker prepare

Build the Docker container for the setup by running the below command

```bash
bash script/start_setup_docker.sh
```

Start the Docker container for the setup by running the below command

```
docker exec -it mlperf_setup bash
```

### Download LLaMA-3.1 405B Instruct Model and prune layers

Go to the root folder by running `cd /`

Access your HuggingFace token and run the following command

```
# Generate an access token on HuggingFace and set it here
export HF_TOKEN="hf_your_access_token"
```

Download the `Llama-3.1-405B-Instruct` model and `Llama-3.1-405B-Instruct-mxfp4` by running the script below

```bash
bash script/download_model.sh
```

Use the following script to prune `[62, 103]` layers of the `Llama-3.1-405B-Instruct` model 

```python
python script/prune_llama.py
```

The pruned model is saved at `model/MLPerf-Pruned-Llama-3.1-405B-Instruct`

### Download train and validation dataset

Set up the environment:

```
cd /RULER/docker
pip install -r requirements.txt
```
Donwload the data

```
cd /RULER/scripts/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh
```

Generate the first part of the data

```
cd /RULER/scripts
bash run.sh llama3.1-405b-mxfp4 synthetic
```

Go to `/RULER/scripts/synthetic.yaml` and modify  `num_needle_v` in `niah_multivalue` from 8 to `16`. Change `RESULTS_DIR` in `run.sh` from `RESULTS_DIR="/data/original_8/${MAX_SEQ_LENGTH}"` to `RESULTS_DIR="/data/original_16/${MAX_SEQ_LENGTH}"`, and run `run.sh` again to generate the second part of the data

```
bash run.sh llama3.1-405b-mxfp4 synthetic
```

Go to `/RULER/scripts/synthetic.yaml` and modify `num_needle_v` in `niah_multivalue` to `32`. Change `RESULTS_DIR` in `run.sh` from `RESULTS_DIR="/data/original_16/${MAX_SEQ_LENGTH}"` to `RESULTS_DIR="/data/original_32/${MAX_SEQ_LENGTH}"`, and run `run.sh` again to generate the third part of the data

```
bash run.sh llama3.1-405b-mxfp4 synthetic
```

After generating three parts of the data, run the following script to combine a total of 9 .jsonl files and prepare them into `train.json` and `valid.json` files for the training in LLaMA-Factory

```
python combine_data.py
```

The two .json files should be in the `/data` folder.



## Fine-tuning

### Docker prepare

Exit the previous docker. Build the Docker container for the LoRA fine-tuning by running the below command

```bash
bash script/start_finetune_docker.sh
```

Start the Docker container for the LoRA fine-tuning by running the below command

```
docker exec -it mlperf_finetune bash
```

Environment setup inside the docker

```
cd /LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip install liger-kernel
pip install deepspeed==0.16.9
```

### LoRA Fine-tuning

Finish the LoRA fine-tuning process and get the LoRA adapter by running the below command

```
llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml
```

After the training is done, merge the LoRA adapter into the original model and get the fine-tuned model by running the below command

```
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

You can modify the `adapter_name_or_path` in `examples/merge_lora/llama3_lora_sft.yaml` with the path of the middle checkpoint for merging. The fine-tuned model is saved at `/model/MLPerf-Pruned-Llama-3.1-405B-Instruct-lora-sft`

Specifically, we use the checkpoint at step 140 for inference, which means that we change the `adapter_name_or_path` into ` saves/llama3-405b_prune_finetune/lora/sft/checkpoint-140`.

### Quantize the fine-tuned model into MXFP4

Exit the previous docker. Start the Docker container for the quantization by running the below command

```
docker exec -it mlperf_setup bash
```

Install MLCommons MLC Automation framework and download MLPerf Calibration dataset

```
pip install mlc-scripts
mlcr get,dataset,mlperf,inference,llama3,_calibration --outdirname=./ -j
```

Finish the quantization process by running the script

```
python /quantization/mxfp4_quantization.py
```

Copy the tokenizer by running the below command

```
cp /model/MLPerf-Pruned-Llama-3.1-405B-Instruct-lora-sft/tokenizer* /model/MLPerf-Pruned-Llama-3.1-405B-Instruct-lora-sft-quantize/
```

The quantized model is saved at `/model/MLPerf-Pruned-Llama-3.1-405B-Instruct-lora-sft-quantize`
