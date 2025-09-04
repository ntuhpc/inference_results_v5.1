# NVIDIA MLPerf Inference Benchmarks

## List of Benchmarks

Please refer to the `README.md` in each benchmark directory for implementation details.
- [llama3.1-405b](llama3.1-405b/tensorrt/README.md)
- [llama3.1-8b](llama3.1-8b/tensorrt/README.md)
- [rgat](rgat/tensorrt/README.md)
- [whisper](whisper/tensorrt/README.md)

## Other Directories

- [common](common) - holds shared scripts to generate TensorRT optimized plan files and to run the harnesses.
- [harness](harness) - holds source codes of the harness interfacing with LoadGen.
- [plugin](plugin) - holds source codes of TensorRT plugins used by the benchmarks.
