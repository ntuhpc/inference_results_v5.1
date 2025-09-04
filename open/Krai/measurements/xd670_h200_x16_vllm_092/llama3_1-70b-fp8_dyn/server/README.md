
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xd670_h200_x16_vllm_092 - llama3_1-70b-fp8_dyn - server

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=16,num_gpus=16,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.95,openai_max_connections=1000,model_path=/nas/users/e63604/work_collection/downloaded_Llama-3.1-70b-Instruct_model,loadgen_target_qps=200,num_openai_workers=31,max_num_seqs=909,max_num_batched_tokens=25162,node_hosts:=sith7:sith8,collection_name=results_280725,sut_name=xd670_h200_x16_vllm_092
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=16,num_gpus=16,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.95,openai_max_connections=1000,model_path=/nas/users/e63604/work_collection/downloaded_Llama-3.1-70b-Instruct_model,loadgen_target_qps=200,num_openai_workers=31,max_num_seqs=909,max_num_batched_tokens=25162,node_hosts:=sith7:sith8,collection_name=results_280725,sut_name=xd670_h200_x16_vllm_092
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=16,num_gpus=16,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.95,openai_max_connections=1000,model_path=/nas/users/e63604/work_collection/downloaded_Llama-3.1-70b-Instruct_model,loadgen_target_qps=200,num_openai_workers=31,max_num_seqs=909,max_num_batched_tokens=25162,node_hosts:=sith7:sith8,loadgen_compliance_test=TEST06,collection_name=results_280725,sut_name=xd670_h200_x16_vllm_092
```

