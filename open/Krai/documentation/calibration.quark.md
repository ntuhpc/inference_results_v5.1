# Clone the repo:
```
git clone [https://github.com/mlcommons/inference_results_v5.0](https://github.com/mlcommons/inference_results_v5.0)
```
# Build a docker image for model quantization
```
cd inference_results_v5.0/closed/AMD
bash setup/build_model_and_dataset_env.sh
```
Ignore the warning:
<pre>
 1 warning found (use docker --debug to expand):
 - InvalidDefaultArgInFrom: Default value for ARG $BASE_IMAGE results in empty or invalid base image name (line 2)
</pre>
# Start the container for model quantiation
```
export LAB_MODEL=<your prefered model path>
export LAB_DATASET=<your prefered dataset path>
bash setup/start_model_and_dataset_env.sh
```
# Inside of the container download the model
```
HUGGINGFACE_ACCESS_TOKEN="<your HF token goes here>" python download_model.py
```
# Inside of the container download the dataset
```
bash download_llama2_70b.sh
```
# Inside of the container quantize the model
```
bash quantize_llama2_70b.sh
```
# The result
The quantized model will be available at 
<pre>
${LAB_MODEL}/fp8_quantized
</pre>