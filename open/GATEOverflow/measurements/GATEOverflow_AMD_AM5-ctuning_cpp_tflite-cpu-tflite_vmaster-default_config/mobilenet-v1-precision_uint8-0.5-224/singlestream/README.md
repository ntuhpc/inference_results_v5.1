*Check [MLC MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-6.14.0-24-generic-x86_64-with-glibc2.41
* CPU version: x86_64
* Python version: 3.13.3 (main, Jun 16 2025, 18:15:32) [GCC 14.2.0]
* MLC version: unknown

## MLC Run Command

See [MLC installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo gateoverflow@mlperf-automations --checkout=82cd4d77c9d362e147c0867afeeb660423290ebe


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload gateoverflow@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo gateoverflow@mlperf-automations
mlc pull repo gateoverflow@mlperf-automations
mlc rm cache -f

```

## Results

Platform: GATEOverflow_AMD_AM5-ctuning_cpp_tflite-cpu-tflite_vmaster-default_config

Model Precision: uint8

### Accuracy Results 
`acc`: `62.308`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`90th percentile latency (ns)`: `2031765.0`
