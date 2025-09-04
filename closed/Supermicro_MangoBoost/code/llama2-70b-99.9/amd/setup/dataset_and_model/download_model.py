# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, os, sys

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="/model", type=str, help="Path to the model location")
args = parser.parse_args()
token=os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")

if token == "":
    print("Missing HUGGINGFACE_ACCESS_TOKEN")
    sys.exit(1)

models = (
    ("meta-llama/Llama-2-70b-chat-hf", "llama2-70b-chat-hf/orig"),
    # Premade models (requires token)
    # ("amd/Llama-2-70b-chat-hf_FP8_MLPerf_V2", "llama2-70b-chat-hf/fp8_quantized"),
)

for model_id, model_path in models:
    # Create the directory
    download_dir = f"{args.model_dir}/{model_path}"
    os.makedirs(download_dir,  exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        local_dir=download_dir,
        max_workers=32,
        token=token,
    )
    print(f"Succesfully downloaded {model_id}")
