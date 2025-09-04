# ResNet-50 (PyTorch) INT8 Quantization for OpenVINO

## LEGAL DISCLAIMER
To the extent that any data, datasets, or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality. By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license. 

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets, or models. 

## Add the path to your dataset and pytorch weights
Change the `IMAGENET_ROOT` and `MODEL_PATH` directories to match your local system. For example:

```
IMAGENET_ROOT = "/home/user/datasets/imagenet-packages/"
MODEL_PATH = "/home/user/networks/resnet50/pytorch/resnet50-19c8e357.pth"
```

## Set up NNCF virtual environment
Create a Python virtual environment and install the required packages.

```
python3 -m venv nncf_env
source nncf_env/bin/activate
pip install --upgrade pip
pip intall -r ./requirements.txt
```

## Run the quantization script
```
python3 ./resnet50_int8_quantization_nncf.py
```
