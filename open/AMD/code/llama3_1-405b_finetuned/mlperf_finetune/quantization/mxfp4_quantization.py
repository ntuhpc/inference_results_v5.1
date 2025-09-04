import json
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from quark.torch.quantization.config.config import Config
from quark.torch import ModelQuantizer
from quark.torch.export.api import ModelExporter
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig

# --- Configuration ---
MODEL_PATH = "/model/MLPerf-Pruned-Llama-3.1-405B-Instruct-lora-sft"
QUANT_CONFIG_PATH = "./config.json"
OUTPUT_DIR = "/model/MLPerf-Pruned-Llama-3.1-405B-Instruct-lora-sft-quantize"


def _prepare_data_from_pkl(pkl_file_path, tokenizer=None, max_samples=128):
    """Prepares a calibration dataloader from a pickle file using question column."""
    print(f"Loading calibration data from: {pkl_file_path}")
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    
    if not hasattr(data, 'columns') or 'question' not in data.columns:
        raise ValueError("Expected pandas DataFrame with 'question' column")
    
    # Use question column and limit to max_samples
    questions = data['question'].head(max_samples).tolist()
    print(f"Using {len(questions)} questions for calibration")
    
    if tokenizer is None:
        raise ValueError("Tokenizer is required for text data")
    
    # Tokenize questions
    tokenized = tokenizer(questions, truncation=True, padding=True, return_tensors="pt", max_length=1024)
    
    from datasets import Dataset
    dataset = Dataset.from_dict({'input_ids': tokenized['input_ids'].tolist()})
    dataset.set_format(type='torch')
    return DataLoader(dataset, batch_size=4)

def _prepare_data(tokenizer, use_pkl=False, pkl_file_path=None):
    """Prepares a calibration dataloader."""
    if use_pkl:
        return _prepare_data_from_pkl(pkl_file_path, tokenizer)

    dataset = load_dataset('wikitext', 'wikitext-2-v1', split="train", token='') # Your HF tokens here.
    dataset = dataset.shuffle(seed=42).select(range(128))

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return DataLoader(dataset, batch_size=4)

def main():
    """
    Main function to run the quantization process.
    """
    # 1. Load Model and Tokenizer
    # Using bfloat16 for better performance on compatible hardware
    print(f"Loading model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("Model loaded successfully.")

    # 2. Load Quantization Configuration from JSON
    print(f"Loading quantization config from: {QUANT_CONFIG_PATH}")
    with open(QUANT_CONFIG_PATH, 'r') as f:
        config_data = json.load(f)

    # The 'quantization_config' from the model's config.json is the input for Quark's Config object
    quant_config_dict = config_data.get("quantization_config")
    if quant_config_dict is None:
        raise ValueError(f"'quantization_config' not found in {QUANT_CONFIG_PATH}")

    # Create the main Config object from the dictionary
    quant_config = Config.from_dict(quant_config_dict)
    print("Quantization config loaded successfully.")

    # 3. Prepare Calibration Data
    print("Preparing calibration dataloader...")
    # Set use_json to True to load from a custom JSON file
    # and provide the path to your JSON file.
    use_pkl_data = True
    pkl_data_path = "/app/vllm/checkpoint/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl" # MLPerf calibration dataset here. You can check https://docs.mlcommons.org/inference/benchmarks/language/get-llama3_1-405b-data/#__tabbed_1_2 to download the dataset.
    calib_dataloader = _prepare_data(tokenizer, use_pkl=use_pkl_data, pkl_file_path=pkl_data_path)
    print("Calibration dataloader prepared.")

    # 4. Quantize the Model
    print("Starting model quantization...")
    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, dataloader=calib_dataloader)
    print("Model quantization finished.")

    # 5. Export the Quantized Model
    print(f"Exporting quantized model to: {OUTPUT_DIR}")
    export_config = ExporterConfig(
       json_export_config=JsonExporterConfig(weight_format="real_quantized")
    )
    model_exporter = ModelExporter(
       config=export_config,
       export_dir=OUTPUT_DIR
    )
    model_exporter.export_safetensors_model(
       model=quant_model,
       quant_config=quant_config
    )
    print("Model exported successfully.")
    print(f"Find the quantized model and config at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

