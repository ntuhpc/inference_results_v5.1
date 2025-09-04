
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xd670_h200_x24_vllm_092 - llama3_1-70b-fp8_dyn - offline

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=24,num_gpus=24,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.95,openai_max_connections=1000,model_path=/nas/users/e63604/work_collection/downloaded_Llama-3.1-70b-Instruct_model,loadgen_target_qps=100,max_retry_allowed=20,collection_name=results_310725,openai_retry_delay_ms=30000,num_openai_workers=59,max_num_seqs=2401,max_num_batched_tokens=30574,sut_name=xd670_h200_x24_vllm_092
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=24,num_gpus=24,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.95,openai_max_connections=1000,model_path=/nas/users/e63604/work_collection/downloaded_Llama-3.1-70b-Instruct_model,loadgen_target_qps=300,n_trials=100,max_retry_allowed=20,num_check_new_max=0,collection_name=experiments,study_name=offline_300725,openai_retry_delay_ms=30000,num_openai_workers=59,max_num_seqs=2401,max_num_batched_tokens=30574
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=24,num_gpus=24,quantization=fp8,max_seq_len_to_capture=1024,gpu_memory_utilization=0.95,openai_max_connections=1000,model_path=/nas/users/e63604/work_collection/downloaded_Llama-3.1-70b-Instruct_model,loadgen_target_qps=300,max_retry_allowed=20,collection_name=results_310725,openai_retry_delay_ms=30000,num_openai_workers=59,max_num_seqs=2401,max_num_batched_tokens=30574,loadgen_compliance_test=TEST06
```

