
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xd670_h200_x8_sglang_049 - llama3_1-70b-fp8_pre - offline

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,inference_server=sglang,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,quantization=None,docker_network=krai_test,model_path=/nas/users/e63604/work_collection/downloaded_Meta-Llama-3.1-70B-Instruct-FP8_model,loadgen_target_qps=100,mem_fraction_static=0.9,collection_name=results_210725_pre,attention_backend=fa3,num_openai_workers=95,openai_max_connections=102,max_running_requests=1047,sut_name=xd670_h200_x8_sglang_049
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,inference_server=sglang,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,quantization=None,docker_network=krai_test,model_path=/nas/users/e63604/work_collection/downloaded_Meta-Llama-3.1-70B-Instruct-FP8_model,loadgen_target_qps=100,mem_fraction_static=0.9,collection_name=results_210725_pre,attention_backend=fa3,num_openai_workers=95,openai_max_connections=102,max_running_requests=1047,sut_name=xd670_h200_x8_sglang_049
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama2,framework=openai,inference_server=sglang,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,quantization=None,docker_network=krai_test,model_path=/nas/users/e63604/work_collection/downloaded_Meta-Llama-3.1-70B-Instruct-FP8_model,loadgen_target_qps=100,mem_fraction_static=0.9,collection_name=results_210725_pre,attention_backend=fa3,num_openai_workers=95,openai_max_connections=102,max_running_requests=1047,sut_name=xd670_h200_x8_sglang_049,loadgen_compliance_test=TEST06
```

