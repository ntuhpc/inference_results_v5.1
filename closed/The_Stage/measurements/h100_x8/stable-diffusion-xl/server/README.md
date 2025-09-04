
# MLPerf Inference v5.0 - Closed - The_Stage

To run experiments individually, use the following commands.

## h100_x8 - stable-diffusion-xl - server

### Accuracy  

```
axs byquery loadgen_output,task=text_to_image,framework=stageai,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,loadgen_dataset_size=5000,loadgen_buffer_size=5000,backend=stageai,model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,device=cuda,axs_device_id=0+1+2+3+4+5+6+7,num_gpus=8,dtype=fp16,qps=18,sut_name=h100_x8
```

### Performance 

```
axs byquery loadgen_output,task=text_to_image,framework=stageai,loadgen_scenario=Server,loadgen_mode=PerformanceOnly,loadgen_dataset_size=5000,loadgen_buffer_size=5000,backend=stageai,model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,device=cuda,axs_device_id=0+1+2+3+4+5+6+7,num_gpus=8,dtype=fp16,qps=18,sut_name=h100_x8,count=12036,loadgen_target_qps=None
```

### Compliance TEST01

```
axs byquery loadgen_output,task=text_to_image,framework=stageai,loadgen_scenario=Server,loadgen_mode=PerformanceOnly,loadgen_dataset_size=5000,loadgen_buffer_size=5000,backend=stageai,model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,device=cuda,axs_device_id=0+1+2+3+4+5+6+7,num_gpus=8,dtype=fp16,qps=18,sut_name=h100_x8,count=12036,loadgen_compliance_test=TEST01,loadgen_target_qps=None
```

### Compliance TEST04

```
axs byquery loadgen_output,task=text_to_image,framework=stageai,loadgen_scenario=Server,loadgen_mode=PerformanceOnly,loadgen_dataset_size=5000,loadgen_buffer_size=5000,backend=stageai,model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,device=cuda,axs_device_id=0+1+2+3+4+5+6+7,num_gpus=8,dtype=fp16,qps=18,sut_name=h100_x8,count=12036,loadgen_compliance_test=TEST04,loadgen_target_qps=None
```

