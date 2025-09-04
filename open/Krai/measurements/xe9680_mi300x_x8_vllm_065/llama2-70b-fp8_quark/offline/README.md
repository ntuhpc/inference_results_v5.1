
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xe9680_mi300x_x8_vllm_065 - llama2-70b-fp8_quark - offline

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,backend=rocm,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,model_path=/mnt/llm_data/krai/models/mlperf5.0_AMD/llama2-70b-chat-hf/fp8_quantized,block_size=32,num_loadgen_workers=1,server_docker_image=mlperf_inference_submission,server_docker_image_tag=5.0,num_openai_workers=59,openai_max_connections=199,max_num_batched_tokens=38890,max_num_seqs=1275,gpu_memory_utilization=0.96,loadgen_target_qps=95,loadgen_min_duration_s=14400,docker_network=axs,chat_template+
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,backend=rocm,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,model_path=/mnt/llm_data/krai/models/mlperf5.0_AMD/llama2-70b-chat-hf/fp8_quantized,block_size=32,num_loadgen_workers=1,server_docker_image=mlperf_inference_submission,server_docker_image_tag=5.0,num_openai_workers=59,openai_max_connections=199,max_num_batched_tokens=38890,max_num_seqs=1275,num_scheduler_steps=12,gpu_memory_utilization=0.96,loadgen_target_qps=95,loadgen_min_duration_s=14400,docker_network=axs,chat_template+,collection_name=experiments_quark
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,backend=rocm,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,model_path=/mnt/llm_data/krai/models/mlperf5.0_AMD/llama2-70b-chat-hf/fp8_quantized,block_size=32,num_loadgen_workers=1,server_docker_image=mlperf_inference_submission,server_docker_image_tag=5.0,num_openai_workers=59,openai_max_connections=199,max_num_batched_tokens=38890,max_num_seqs=1275,num_scheduler_steps=12,gpu_memory_utilization=0.96,loadgen_target_qps=95,loadgen_min_duration_s=14400,docker_network=axs,chat_template+,collection_name=experiments_quark,loadgen_compliance_test=TEST06
```

