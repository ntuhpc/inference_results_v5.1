#!/bin/bash

export OPENVINO_DEVICE=GPU
python3 python/main.py --profile resnet50-openvino --model ./mlperf-models/resnet50_v1.onnx --dataset-path ./fake_imagenet/ --output output/cmdline-test-output-singlestream --scenario SingleStream --max-batchsize=1 --time=10
