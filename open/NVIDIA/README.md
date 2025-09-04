# MLPerf Inference - LLaMA3 8B Configuration

## Model Information
- **Model**: LLaMA3.1 8B
- **Submission Type**: SingleStream
- **Hardware**: NVIDIA Thor

## Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Batch Size** | 1 | Number of samples processed simultaneously |
| **Precision** | FP4 | Model weight precision (4-bit floating point) |
| **Quantization Method** | W4A16 AWQ | Weights: 4-bit, Activations: 16-bit using AWQ method |

## Technical Details

### Quantization Strategy
- **AWQ (Activation-aware Weight Quantization)**: Advanced quantization technique that preserves model accuracy while reducing memory footprint
- **W4A16**: 4-bit weights with 16-bit activations for optimal performance-accuracy balance

### Performance Characteristics
- Optimized for single-stream inference scenarios
- Reduced memory usage through 4-bit weight quantization
- Maintained accuracy through activation-aware quantization approach