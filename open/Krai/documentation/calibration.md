# MLPerf Inference v5.1 - Krai

## Calibration (Quantization) Details

Krai submissions on NVIDIA H100/H200/B200 and AMD MI300X GPUs use:
1. fp8 models dynamically quantized with vLLM and SGLang (`llama3_1-70b-fp8_dyn`);
1. an fp8 model pre-quantized by Neural Magic (`llama3_1-70b-fp8_pre`);
1. an fp8 model quantized using NVIDIA's [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (`llama3_1-70b-fp8_trtllm`);
1. an fp8 model quantized using AMD's [Quark Model Optimizer](https://github.com/amd/quark) (`llama2-70b-fp8_quark`).

Model name | Model URL<br>(`starting_weights_filename`) | Weights data type<br>(`weight_data_types`) | Weights transformations<br>(`weight_transformations`)
-|-|-|-
`llama3_1-70b-fp8_dyn` | [https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) | fp16 | fp8 quantization (dynamic using vLLM or SGLang)
`llama3_1-70b-fp8_pre` | [https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8](https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8) | fp8 | none
`llama3_1-70b-fp8_trtllm` | [https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) | fp16 | fp8 quantization ([using modelopt](calibration.modelopt.md))
`llama2-70b-fp8_quark` | [https://huggingface.co/meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)  | fp16 | fp8 quantization ([using quark](calibration.quark.md))
