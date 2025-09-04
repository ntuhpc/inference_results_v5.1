
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xe9680_h100_x8_vllm_092 - llama3_1-70b-fp8_dyn - offline

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=2,pp=1,dp=4,num_gpus=8,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.9,docker_network=axs,model_path=/mnt/nvme3/krai/models/Llama3.1-70B-Instruct,loadgen_target_qps=100,collection_name=experiments_new,sut_name=xe9680_h100_x8_vllm_092,server_docker_image=vllm/vllm-openai,server_docker_image_tag=v0.9.2,num_openai_workers=65,openai_max_connections=76,max_num_seqs=3264,max_num_batched_tokens=14768
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=2,pp=1,dp=4,num_gpus=8,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.9,docker_network=axs,model_path=/mnt/nvme3/krai/models/Llama3.1-70B-Instruct,loadgen_target_qps=100,collection_name=experiments_new,sut_name=xe9680_h100_x8_vllm_092,server_docker_image=vllm/vllm-openai,server_docker_image_tag=v0.9.2,num_openai_workers=65,openai_max_connections=76,max_num_seqs=3264,max_num_batched_tokens=14768
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=2,pp=1,dp=4,num_gpus=8,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.9,docker_network=axs,model_path=/mnt/nvme3/krai/models/Llama3.1-70B-Instruct,loadgen_target_qps=100,collection_name=experiments_new,sut_name=xe9680_h100_x8_vllm_092,server_docker_image=vllm/vllm-openai,server_docker_image_tag=v0.9.2,num_openai_workers=65,openai_max_connections=76,max_num_seqs=3264,max_num_batched_tokens=14768,loadgen_compliance_test=TEST06
```

