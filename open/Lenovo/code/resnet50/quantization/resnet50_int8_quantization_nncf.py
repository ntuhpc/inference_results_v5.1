import json, random, re, subprocess
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import torchvision
from fastdownload import FastDownload
from rich.progress import track
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms

import nncf

ROOT = Path(__file__).parent.resolve()

# Download the devkit and validation dataset to this target directory
IMAGENET_ROOT = "/replace/with/path/to/imagenet/packages"
MODEL_PATH = "/replace/with/path/to/mlperf/resnet/pytorch/checkpoint"

# Load Imagenet Dataset
normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225])

val_dataset = datasets.ImageNet(
    root=IMAGENET_ROOT,
    split="val",
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=14,
)

# Load PyTorch checkpoints from MLPerf into OpenVINO
torch_model = torchhvision.models.resnet50(pretrained=False)
checkpoint = torch.load(MODEL_PATH)
torch_model.load_state_dict(checkpoint)
torch_model.eval()
ov_model = ov.convert_model(torch_model, example_input=torch.randn(1,3,224,224))

# Quantize into INT8
def transform_fn(data_item):
    images, _ = data_item
    return images

calib_indices = random.sample(range(len(val_dataset)),k=512)
calib_subset = torch.utils.data.Subset(val_dataset, calib_indices)
calib_loader = torch.utils.data.DataLoader(
    calib_subset,
    batch_size=16,
    shuffle=False,
    num_workers=14,
)

calib_dataset = nncf.Dataset(calib_loader, transform_fn)
ov_int8_model = nncf.quantize(ov_model, calib_dataset)

# Utility
def validate(model: ov.Model, loader) -> float:
    preds, refs = [], []
    compiled = ov.compile_model(model, "CPU")
    out = compiled.outputs[0]

    for images, target in track(loader, description="Validating"):
        logits = compiled(images)[out]
        preds.append(np.argmax(logits, axis=1))
        refs.append(target)

    return accuracy_score(np.concatenate(preds), np.concatenate(refs))

def run_benchmark(model_path: Path, shape: list[int]) -> float:
    cmd = ["benchmark_app", "-m", model_path.as_posix(), "-d", "CPU",
        "-api", "async", "-t", "15", "-shape", str(shape)]
    out = subprocess.check_output(cmd, text=True)
    print(*out.splitlines()[-8:], sep="\n")
    return float(re.search(r"Throughput\: (.+?) FPS", out).group(1))

def get_model_size(ir: Path, unit: str = "Mb") -> float:
    xml, bin_ = ir.stat().st_size, ir.with_suffix(".bin").stat().st_size
    for u in ["bytes", "Kb", "Mb"]:
        if u == unit:
            break
        xml /= 1024
        bin_ /= 1024
    size = xml + bin_
    print(f"Model graph (xml):   {xml:.3f} {unit}")
    print(f"Model weights (bin): {bin_:.3f} {unit}")
    print(f"Model size:          {size:.3f} {unit}")
    return size

# Save model, benchmark, and validate
fp32_path = ROOT / "resnet50_fp32.xml"
ov.save_model(ov_model, fp32_path, compress_to_fp16=False)
fp32_size = get_model_size(fp32_path)

int8_path = ROOT / "resnet50_int8.xml"
nice_name = "softmax_tensor"
out = ov_int8_model.output(0)
out.get_node().set_friendly_name(nice_name)
out.set_names({nice_name})
ov.save_model(ov_int8_model, int8_path)
int8_size = get_model_size(int8_path)

print("\n[Benchmark] FP32: ")
fp32_fps = run_benchmark(fp32_path, [1, 3, 224, 224])

print("\n[Benchmark] INT8: ")
int8_fps = run_benchmark(int8_path, [1, 3, 224, 224])

print("\n[Accuracy] FP32: ")
fp32_top1 = validate(ov_model, calib_loader)
print(f"Top-1: {fp32_top1:.3f}")

print("\n[Accuracy] INT8: ")
int8_top1 = validate(ov_int8_model, calib_loader)
print(f"Top-1: {int8_top1:.3f}")

print("\n [Report]")
print(f"Accuracy drop:       {fp32_top1 - int8_top1:.3f}")
print(f"Compression rate:    {fp32_size / int8_size:.3f}")
print(f"Throughput speed-up: {int8_fps / fp32_fps:.3f}")
