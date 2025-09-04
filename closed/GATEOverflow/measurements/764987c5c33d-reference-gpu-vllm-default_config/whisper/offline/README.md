*Check [MLC MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-6.14.0-24-generic-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, May 27 2025, 17:12:29) [GCC 11.4.0]
* MLC version: unknown

## MLC Run Command

See [MLC installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo mlcommons@mlperf-automations --checkout=82cd4d77c9d362e147c0867afeeb660423290ebe


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload mlcommons@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo mlcommons@mlperf-automations
mlc pull repo mlcommons@mlperf-automations
mlc rm cache -f

```

## Results

Platform: 764987c5c33d-reference-gpu-vllm-default_config

Model Precision: fp32

### Accuracy Results 
`WER`: `97.95483`, Required accuracy for closed division `>= 2.04643`

### Performance Results 
`Samples per second`: `104.953`
