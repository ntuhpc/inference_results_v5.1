
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xe9680_mi300x_x8_vllm_091 - llama3_1-70b-fp8_dyn - offline

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,backend=rocm,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,model_path=/mnt/llm_data/krai/models/Llama-3.1-70B-Instruct,block_size=32,server_docker_image=rocm/vllm,server_docker_image_tag=rocm6.4.1_vllm_0.9.1_20250715,loadgen_target_qps=95,gpu_memory_utilization=0.96,docker_network=axs,n_trials=10,max_retry_allowed=5,num_check_new_max=0,collection_name=experiments,study_name=offline_240725,num_openai_workers=67,openai_max_connections=83,max_num_batched_tokens=17338,max_num_seqs=1866
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,backend=rocm,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,model_path=/mnt/llm_data/krai/models/Llama-3.1-70B-Instruct,block_size=32,server_docker_image=rocm/vllm,server_docker_image_tag=rocm6.4.1_vllm_0.9.1_20250715,loadgen_target_qps=95,gpu_memory_utilization=0.96,docker_network=axs,n_trials=10,max_retry_allowed=5,num_check_new_max=0,collection_name=experiments,study_name=offline_240725,num_openai_workers=67,openai_max_connections=83,max_num_batched_tokens=17338,max_num_seqs=1866
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,backend=rocm,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,model_path=/mnt/llm_data/krai/models/Llama-3.1-70B-Instruct,block_size=32,server_docker_image=rocm/vllm,server_docker_image_tag=rocm6.4.1_vllm_0.9.1_20250715,loadgen_target_qps=95,gpu_memory_utilization=0.96,docker_network=axs,n_trials=10,max_retry_allowed=5,num_check_new_max=0,collection_name=experiments,study_name=offline_240725,num_openai_workers=67,openai_max_connections=83,max_num_batched_tokens=17338,max_num_seqs=1866,loadgen_compliance_test=TEST06
```

