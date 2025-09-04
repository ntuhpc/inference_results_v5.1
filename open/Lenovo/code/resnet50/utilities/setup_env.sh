#!/bin/bash

python3 -m venv mlperf_env
source mlperf_env/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

cd ../../loadgen
CFLAGS="-std=c++14" python setup.py install

cd ../vision/classification_and_detection

mkdir mlperf-models
wget https://zenodo.org/record/4735647/files/resnet50_v1.onnx
wget https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx
wget https://zenodo.org/record/3157894/files/mobilenet_v1_1.0_224.onnx

mv ./res*.onnx mlperf-models/.
mv ./mobile*.onnx mlperf-models/.

./tools/make_fake_imagenet.sh

cd ./tools
./openimages_mlperf.sh -d openimages-128subsample -m 128
mv openimages-128subsample ../.
cd ../.

python3 -m pip uninstall -y onnxruntime-openvino
python3 -m pip install onnxruntime-openvino
echo "[Info] MLPerf Environment created. Please source mlperf_env/bin/activate to begin."
