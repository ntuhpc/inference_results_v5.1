
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xd670_h200_x8_vllm_092 - llama3_1-70b-fp8_dyn - offline

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.95,docker_network=krai_test,model_path=/nas/users/e63604/work_collection/downloaded_Llama-3.1-70b-Instruct_model,collection_name=results_110725,num_openai_workers=91,openai_max_connections=125,max_num_seqs=1232,max_num_batched_tokens=8121,sut_name=xd670_h200_x8_vllm_092
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.95,docker_network=krai_test,model_path=/nas/users/e63604/work_collection/downloaded_Llama-3.1-70b-Instruct_model,loadgen_target_qps=100,collection_name=results_110725,num_openai_workers=91,openai_max_connections=125,max_num_seqs=1232,max_num_batched_tokens=8121,sut_name=xd670_h200_x8_vllm_092
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.95,docker_network=krai_test,model_path=/nas/users/e63604/work_collection/downloaded_Llama-3.1-70b-Instruct_model,loadgen_target_qps=100,collection_name=results_110725,num_openai_workers=91,openai_max_connections=125,max_num_seqs=1232,max_num_batched_tokens=8121,sut_name=xd670_h200_x8_vllm_092,loadgen_compliance_test=TEST06
```

