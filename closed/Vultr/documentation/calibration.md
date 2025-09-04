## AMD MLPerf Inference Calibration and Quantization Details

We utilize the [AMD Quark](https://quark.docs.amd.com/latest/) framework to quantize models for the AMD MLPerf Inference submission. Specifically, these models are quantized to OCP FP8-e4m3 using AMD Quark through the following approach.

## Calibration Dataset and Pre-process

We utilized the entire calibration dataset provided by mlcommons/inference for each model.

The input of calibration dataset was serialized and systematically processed into fixed-length sequences through tokenization, incorporating dynamic padding and truncation as part of the preprocessing step.

## FP8 Quantization Strategy

### FP8 Quantization Process:

We apply per-tensor symmetric static quantization for OCP FP8-e4m3 to quantize all models. The quantized value xq is computed from the original value x as:

x_q = round( clip (x / scale * 448, -448, 448))

where “scale” is the maximum absolute value (amax) of x, 448 represents the range of OCP FP8-e4m3, and scaled value is rounded using the half-even method after clipping.

### Quantization Layers:

All nn.Linear modules, including inputs and weights, within the model decoder block are quantized. The quantization scales for inputs and weights are determined statically, meaning they are fully computed during the calibration step before inference. The weights of Q, K, and V share a common quantization scale, which is set to the maximum of their individual scales. To minimize quantization error for Q, K, and V, we employ the AutoSmoothQuant algorithm in AMD Quark. KV cache entries are also quantized, with K and V sharing a quantization scale that is computed during calibration. For certain models, we scale the KV cache to 1.0, which helps mitigate overfitting in quantized models.

Summarizing the quantized tensors and key configurations:

#### LLaMA-2-70B

* The inputs and weights of linear modules within the decoder blocks are quantized.
* The quantization scales for inputs and weights are computed during calibration.
* Q, K, and V share a common weight quantization scale.
* KV cache entries are quantized, with K and V sharing a common quantization scale determined during calibration.

#### Mixtral-8x7B

* The inputs and weights of linear modules within the decoder blocks are quantized.
* The quantization scales for inputs and weights are computed during calibration.
* KV cache entries are quantized and scaled to 1.0, with K and V sharing a common quantization scale determined during calibration.
