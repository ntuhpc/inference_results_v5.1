
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xe9680_mi300x_x8_vllm_091 - llama3_1-70b-fp8_dyn - server

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_openai_workers=32,num_loadgen_workers=1,backend=rocm,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,model_path=/mnt/llm_data/krai/models/Llama-3.1-70B-Instruct,max_num_seqs=1866,max_seq_len_to_capture=1024,max_num_batched_tokens=17338,gpu_memory_utilization=0.99,loadgen_target_qps=55,openai_client_max_retries=0,openai_max_connections=384,openai_max_keepalive_connections=384,server_docker_image=rocm/vllm,server_docker_image_tag=rocm6.4.1_vllm_0.9.1_20250715,collection_name=experiments,study_name=server_250725,docker_network=axs
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_openai_workers=32,num_loadgen_workers=1,backend=rocm,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,model_path=/mnt/llm_data/krai/models/Llama-3.1-70B-Instruct,max_num_seqs=1866,max_seq_len_to_capture=1024,max_num_batched_tokens=17338,gpu_memory_utilization=0.99,loadgen_target_qps=55,openai_client_max_retries=0,openai_max_connections=384,openai_max_keepalive_connections=384,server_docker_image=rocm/vllm,server_docker_image_tag=rocm6.4.1_vllm_0.9.1_20250715,collection_name=experiments,study_name=server_250725,docker_network=axs
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_openai_workers=32,num_loadgen_workers=1,backend=rocm,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,model_path=/mnt/llm_data/krai/models/Llama-3.1-70B-Instruct,max_num_seqs=1866,max_seq_len_to_capture=1024,max_num_batched_tokens=17338,gpu_memory_utilization=0.99,loadgen_target_qps=55,openai_client_max_retries=0,openai_max_connections=384,openai_max_keepalive_connections=384,server_docker_image=rocm/vllm,server_docker_image_tag=rocm6.4.1_vllm_0.9.1_20250715,collection_name=experiments,study_name=server_250725,docker_network=axs,loadgen_compliance_test=TEST06
```

