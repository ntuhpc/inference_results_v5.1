**System:** Apple MacBook Pro (M1 Pro 16 GB, 2021)  
**Benchmark:** ResNet-50 v1.5 — FP32 ONNX (`resnet50-v1-12.onnx`)


| Item | Value |
|------|-------|
| Execution provider | **CoreMLExecutionProvider** (`ORT_ENABLE_COREML=1`) |
| GPU/ANE policy     | `ORT_COREML_USE_GPU_ONLY=0` (allow ANE) |
| Inference threads  | `8` |
| Dataset            | 50 000-image ImageNet-1k validation set, pre-processed via `mlcr get,dataset,imagenet,validation,_full` |
| Model checksum     | `SHA-256 <fill-in-checksum>` |

Accuracy ≥ 75.46 % and latency/throughput follow MLPerf Edge rules.
No proprietary code or weights were used.
