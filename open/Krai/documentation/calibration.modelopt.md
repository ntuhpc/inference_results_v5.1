# Download the FP16 HF checkpoint
```
export HF_TOKEN="<your HF token goes here>"
axs byquery downloaded,hf_model,model_family=llama3_1,variant=70b,hf_token=${HF_TOKEN}
```
# Start the container for model quantiation
```
export MODEL_PATH=$(axs byquery downloaded,hf_model,model_family=llama3_1,variant=70b,hf_token=${HF_TOKEN} , get_path)
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network host \
-v ${MODEL_PATH}:/model nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc4
```
# Inside of the container clone the modelopt repo
```
cd ~
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer
```
# Inside of the container quantize the model
```
ROOT_SAVE_PATH=/model ~/TensorRT-Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
--model /model --quant fp8 --tp 1 --export_fmt hf
```
# The result
The quantized model will be available at 
<pre>
${MODEL_PATH}/saved_models_model_fp8_hf
</pre>