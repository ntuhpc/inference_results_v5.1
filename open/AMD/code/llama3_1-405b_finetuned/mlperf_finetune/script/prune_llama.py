import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn as nn
import json

ORIGINAL_MODEL_PATH = "/model/Llama-3.1-405B-Instruct" 
SAVE_PATH = "/model/MLPerf-Pruned-Llama-3.1-405B-Instruct" 
KEEP_LAYERS = list(range(0, 62)) + list(range(104, 126))


def prune_llama_model():
    model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_PATH,
        torch_dtype="auto",
    )

    original_layers = model.model.layers
    pruned_layers = nn.ModuleList([original_layers[i] for i in KEEP_LAYERS])
    model.model.layers = pruned_layers
    model.config.num_hidden_layers = len(pruned_layers)

    print(f"reserved layers: {len(pruned_layers)}")

    print("saving pruned model...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("modifying config.json...")
    config_path = os.path.join(SAVE_PATH, "config.json")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    config_data["num_hidden_layers"] = len(pruned_layers)
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"saved into: {SAVE_PATH}")


if __name__ == "__main__":
    prune_llama_model()