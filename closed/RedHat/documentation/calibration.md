# MLPerf Inference Calibration and Quantization Details
## Red Hat  MLPerf Quantization

Post-training quantization (PTQ) for the model is described here https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8. 

### Weights and Activations

This model was obtained by quantizing the weights and activations of Meta-Llama-3.1-8B-Instruct to FP8 data type, ready for inference with vLLM built from source. This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-tensor quantization is applied, in which a single linear scaling maps the FP8 representations of the quantized weights and activations. LLM Compressor is used for quantization with 512 sequences of UltraChat.

Dynamic range values are generally per-channel (or per-row for matrix multiply). In a few cases, a per-tensor value is used. We find the maximum absolute value `t` of any element of the channel or tensor, and the dynamic range is then `[-t,t]`.

