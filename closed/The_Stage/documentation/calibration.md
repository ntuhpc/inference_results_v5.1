# Quantization details
We employed our framework for automated neural networks acceleration (ANNA) to quantize the UNet component of the SDXL pipeline. ANNA is designed to compress neural network models to meet specified size or performance constraints while minimizing accuracy degradation by keeping the outputs as close as possible to the original model.

For each module in the network ANNA decides whether to apply one of the available quantization algorithms or keep the module in its original FP16 precision.

The bag of quantization algorithms includes three variants of post-training static quantization (PTQ) that convert both weights and activations to 8-bit floating-point format. The variants differ in their percentile-based range estimation:

1. **Conservative clipping**: 1st to 99th percentile (removes 2% of outliers)
2. **Moderate clipping**: 5th to 95th percentile (removes 10% of outliers)  
3. **Aggressive clipping**: 10th to 90th percentile (removes 20% of outliers)

Modules that were used for quantization analysis with ANNA:
- all convolutional modules
- all attention modules (including linear submodules). We quantized attention modules to use FP8 attention kernels for optimized performance.
- all linear modules in mlp blocks

**Weight Quantization**: Applied symmetric per-channel quantization, where each output channel has its own scale factor. The quantization range was determined using the absolute maximum value in channel.

**Activation Quantization**: Applied symmetric per-tensor quantization, where a single scale factor is used for the entire activation tensor.

We collected calibration data by running the SDXL pipeline on a set of calibration prompts and capturing the intermediate activations. The calibration parameters were:
- **Inference steps**: 20 diffusion steps per image
- **Guidance scale**: 8.0
- **height**: 1024
- **width**: 1024

This calibration data was used to estimate quantization scales for all three PTQ variants.

ANNA formulates the quantization configuration selection as an optimization problem with following components:

1. **Decision Variables**: Binary variables `t[i,j]` where:
   - `t[i,j] = 1`: module `i` is quantized using algorithm `j`
   - `t[i,j] = 0` for `j` in range(0, 3) - module `i` is not quantized

2. **Objective Function**: We estimate MSE between outputs of original model and compressed model as quadratic function of variables `t[i,j]`.

3. **Size Constraint**: A linear constraint function approximates the model size based on the quantization configuration `t[i,j]`.

4. **Solver**: The resulting mixed integer quadratic programming problem is solved using dynamic programming.

To explore the trade-off between model size and image generation quality using ANNA we generated several different quantized checkpoints with model size varying from 50% to 60% of the original model size.
Starting from full quantized model (50 %) we have been increasing model size constraint until we found the checkpoint that satisfies requirements on FID and CLIP scores. Quickly enough we found the checkpoint with model size equal to 50.1% meets the requirements. This checkpoint is only 0.1 % larger than the full quantized model, but this buffer of 0.1 % allowed us to keep 10 most important modules in FP16 precision to retain FID and CLIP scores in required range.

For instance, on [coco_cal_captions_list.txt](https://github.com/mlcommons/inference/blob/master/calibration/COCO-2014/coco_cal_captions_list.txt) ANNA have found the following configurtion: 

`conv_in, time_embedding.linear_1down_blocks.0.downsamplers.0.conv
down_blocks.1.resnets.0.conv_shortcut
up_blocks.1.resnets.2.conv_shortcut
up_blocks.2.resnets.0.conv_shortcut
up_blocks.2.resnets.1.conv_shortcut
up_blocks.2.resnets.2.conv2
up_blocks.2.resnets.2.conv_shortcut
conv_out` - use `float16` dtype for inference, while other convolutions, linear and attention layers use `float8` data type.

