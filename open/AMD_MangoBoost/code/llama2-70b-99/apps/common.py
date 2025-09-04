from absl import flags

# Note: The sharegpt dataset has similar format to the llama2 dataset
#       Originally there are 330k questions in the sharegpt dataset

dataset_info_map = {
    "gptj": {
        "dataset_path": "/workspace/data/gptj_data.json",
        "total_sample_count": 13368,
    },
    "llama2": {
        "dataset_path": "/workspace/data/llama2_data.pkl",
        "total_sample_count": 24576,
        "input_key": "input",
        "output_key": "output",
    },
    # llama2 and openorca are exactly the same dataset
    "openorca": {
        "dataset_path": "/workspace/data/llama2_data.pkl",
        "total_sample_count": 24576,
        "input_key": "input",
        "output_key": "output",
    },
    "mixtral-8x7b": {
        "dataset_path": "/workspace/data/mixtral_data.pkl",
        "total_sample_count": 15000,
        "input_key": "input",
        "output_key": "ref_output",
    },
    "sharegpt": {
        "dataset_path": "/workspace/data/sharegpt_50k.pkl",
        "total_sample_count": 50000,
        "input_key": "question",
        "output_key": "output",
    },
    "cnn_dailymail": {
        "dataset_path": None,  # will be downloaded
        "total_sample_count": 13368,  # use the validation set
        "input_key": "question",
        "output_key": "output",
    },
    "squad": {
        "dataset_path": None,  # will be downloaded
        "total_sample_count": 10570,  # use the validation set
        "input_key": "question",
        "output_key": "output",
    },
    "flickr30k": {
        "dataset_path": None,
        "total_sample_count": 31014,  # use the validation set
        "output_key": "output",
    },
    "llava-instruct-mix-vsft": {
        "dataset_path": None,
        "total_sample_count": 259155,  # use the train set
        "output_key": "output",
    },
}

flags.DEFINE_string(
    "model_name",
    "meta-llama/Llama-2-7b-chat-hf",
    "Model name",
)
flags.DEFINE_integer(
    "num_prompts",
    10000,
    "Number of prompts to generate",
)
flags.DEFINE_integer(
    "tp",
    0,
    "tensor parallelism",
)
flags.DEFINE_integer(
    "dp",
    0,
    "number of workers",
)
flags.DEFINE_enum(
    "load_format",
    "auto",
    ["bitsandbytes", "bitsandbytesint8", "auto"],
    "Quantized model to use",
)
flags.DEFINE_enum(
    "quantization",
    "None",
    ["bitsandbytes", "fp8", "None"],
    "Quantization scheme to use",
)
flags.DEFINE_enum(
    "kv_cache_dtype",
    "auto",
    ["auto", "fp8"],
    "KV cache dtype",
)
flags.DEFINE_string(
    "quantization_param_path",
    None,
    "Path to quantization parameters",
)
flags.DEFINE_string(
    "quantization_weight_path",
    None,
    "Path to quantization weights",
)
flags.DEFINE_integer(
    "max_num_seqs",
    256,
    "max number of sequences to process in parallel",
)
flags.DEFINE_integer(
    "max_model_len",
    None,
    "maximum sequence length supported by the engine",
)
flags.DEFINE_integer(
    "block_size",
    None,
    "Token block size for contiguous chunks of tokens in memory",
)
flags.DEFINE_integer(
    "max_num_batched_tokens",
    None,
    "maximum number of tokens that can be processed in a single batch",
)
flags.DEFINE_bool(
    "enforce_eager",
    False,
    "enable eager mode execution",
)
flags.DEFINE_bool(
    "streaming",
    False,
    "benchmark streaming requests",
)
flags.DEFINE_string(
    "output",
    None,
    "Path to output file",
)
flags.DEFINE_bool(
    "force",
    False,
    "Force overwrite of output file",
)


flags.register_validator(
    "num_prompts",
    lambda x: x > 0,
    message="num_prompts must be greater than 0",
)

flags.register_validator(
    "max_model_len",
    lambda x: x is None or x > 0,
    message="max_model_len must be greater than 0",
)
flags.register_validator(
    "block_size",
    lambda x: x is None or x in [8, 16, 32, 64, 128],
    message="Token block size for contiguous chunks of tokens in memory",
)
flags.register_validator(
    "max_num_batched_tokens",
    lambda x: x is None or x > 0,
    message="max_num_batched_tokens must be greater than 0",
)
flags.register_validator(
    "tp",
    lambda x: x > -1 and x < 9,
    message="tp must be between 1 and 8 or 0 for auto mode",
)
flags.register_validator(
    "dp",
    lambda x: x > -1 and x < 9,
    message="dp must be between 1 and 8 or 0 for auto mode",
)
flags.DEFINE_string(
    "load_balancing_mode",
    "auto",
    "Load Balancing method",
)
flags.DEFINE_integer(
    "gpu_batch_size",
    48,
    "The max number of queries to be sent to each worker as a batch.",
)
flags.DEFINE_float(
    "batcher_threshold",
    0.4,
    "The max duration (in sec) that we wait to send the pending queries.",
)
flags.DEFINE_bool(
    "semi_batching",
    False,
    "Per query only send the outputs twice (first token and rest tokerns)",
)
