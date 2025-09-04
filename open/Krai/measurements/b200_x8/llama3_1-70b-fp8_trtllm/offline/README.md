
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## b200_x8 - llama3_1-70b-fp8_trtllm - offline

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,inference_server=trtllm,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,model_path=/home/oleg/work_collection/downloaded_Llama-3.1-70b-Instruct_model/saved_models_model_fp8_hf,loadgen_target_qps=100,collection_name=tests,num_openai_workers=32,openai_max_connections=750,server_docker_image_tag=1.0.0rc4,wait_for_server_timeout_s=1200,max_batch_size=2350
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,inference_server=trtllm,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,model_path=/home/oleg/work_collection/downloaded_Llama-3.1-70b-Instruct_model/saved_models_model_fp8_hf,loadgen_target_qps=220,server_docker_image_tag=1.0.0rc4,wait_for_server_timeout_s=1200,n_trials=100,num_check_new_max=0,collection_name=experiments,study_name=offline_310725,num_openai_workers=29,openai_max_connections=604,max_batch_size=1572
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama2,framework=openai,inference_server=trtllm,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,model_path=/home/oleg/work_collection/downloaded_Llama-3.1-70b-Instruct_model/saved_models_model_fp8_hf,loadgen_target_qps=220,server_docker_image_tag=1.0.0rc4,wait_for_server_timeout_s=1200,collection_name=results_010825,num_openai_workers=29,openai_max_connections=604,max_batch_size=1572,loadgen_compliance_test=TEST06,sut_name=b200_x8
```

