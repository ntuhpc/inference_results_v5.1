# MLPerf Inference v5.1 - Krai

We present LLM submissions using vLLM, SGLang and TensorRT-LLM on:
- HPE Cray XD670 servers with NVIDIA H200 GPUs
- Dell PowerEdge XE9680 servers with NVIDIA H100 GPUs and AMD MI300X GPUs
- Cloud instances with NVIDIA B200 GPUs.

We cordially thank our partners for providing access to computational resources!

The submissions use the [KRAI](https://krai.ai) [KISS](http://github.com/krai/axs2kiss) (KRAI Inference Serving Solution) for fast, efficient and scalable inference, and the [KRAI X](http://github.com/krai/axs) technology for workflow automation.

Detailed setup instructions per workload are provided in README files under the [code](code) directory.
Individual benchmarking commands per system, workload, scenario and mode are provided in README files under the respective [measurements](measurements) directories.
Quantization details are provided under the [documentation](documentation) directory.

The source code has been released under the permissive MIT license across several public repositories (under the `mlperf_5.1` branches created by the v5.1 submission deadline):

- https://github.com/krai/axs (KRAI X Workflow Automation Technology)
- https://github.com/krai/axs2kiss (KRAI Inference Serving Solution)
- https://github.com/krai/axs2mlperf

## Contact

For any queries please contact info@krai.ai.
