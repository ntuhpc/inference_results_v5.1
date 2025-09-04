# Copyright 2024, MangoBoost, Inc. All rights reserved.


import logging
import pickle
import time
import json
import threading
import common
import nltk
import pandas as pd
from PIL import Image
import os
import aiohttp
import asyncio
import random
import numpy as np


from common import dataset_info_map
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from absl import app, flags
from tqdm import tqdm
from typing import List, Dict, Tuple, Union
from datasets import load_dataset

# Get the common flags
flags.adopt_module_key_flags(common)

# Module Specific Flags
flags.DEFINE_enum(
    "dataset",
    "llama2",
    dataset_info_map.keys(),
    "Dataset to use (llama2, sharegpt, etc.)",
)
flags.DEFINE_integer(
    "beam_width",
    0,
    "beam width if used",
)
flags.DEFINE_bool(
    "snippet",
    False,
    "display snippet of the generated text",
)
flags.DEFINE_integer(
    "max_parallel_reqs",
    0,
    "Max number of parallel requests, 0 mean unlimited requests",
)
flags.DEFINE_integer(
    "pps",
    0,
    "prompts per second, 0 means no limit",
)
flags.register_validator(
    "beam_width",
    lambda x: x >= 0,
    message="beam_width must be greater than or equal to 0",
)

# Benchmarking Flags
flags.DEFINE_bool(
    "use_endpoint",
    False,
    "use a web endpoint for benchmarking",
)
flags.DEFINE_string(
    "endpoint",
    "http://localhost:8000/v1/chat/completions",
    "supply an OpenAI-compatible chat endpoint",
)
flags.DEFINE_integer(
    "timeout",
    3600,  # 60 minutes
    "Per‑request HTTP timeout (seconds) for endpoint-based benchmarking",
)
flags.DEFINE_bool(
    "use_http2",
    False,
    "Use HTTP2 instead of HTTP1 for endpoint-based benchmarking",
)
flags.register_validator(
    "timeout",
    lambda x: x > 0,
    message="timeout must be greater than 0",
)
# image benchmarking
flags.DEFINE_enum(
    "query_type",
    "text",
    ["text", "image"],
    "Input format (e.g. text or image)",
)
flags.DEFINE_integer(
    "max_image_per_prompt", 1, "How many images to be merged into one prompt."
)
flags.register_validator(
    "max_image_per_prompt",
    lambda x: x >= 1,
    message="max_image_per_prompt must be >= 0",
)
flags.DEFINE_string(
    "dataset_path",
    None,
    "Path to the dataset",
)

FLAGS = flags.FLAGS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Accuracy")


def _manual_chat_format(chat: List[Dict[str, str]]) -> str:
    formatted_chat = ""
    for message in chat:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            formatted_chat += f"[SYSTEM]: {content}\n"
        elif role == "user":
            formatted_chat += f"[USER]: {content}\n"
        elif role == "assistant":
            formatted_chat += f"[ASSISTANT]: {content}\n"
        else:
            formatted_chat += f"[{role.upper()}]: {content}\n"
    return formatted_chat.strip()


def apply_format(
    tokenizer: PreTrainedTokenizerBase,
    chat: Union[str, Dict[str, str], List[Dict[str, str]]],
) -> Dict[str, Union[str, int]]:
    """
    Apply the model specific format to the chat.

    Emulates LLMBoost's apply_format function to maintain compatibility.

    Args:
        chat (str, Dict, List[Dict]): The chat to be formatted.
        model_name (str): The name of the model to format for.

    Returns:
        Dict: The formatted chat with id and val keys.
    """
    # Validate the chat
    if isinstance(chat, str):
        chat = [{"role": "user", "content": chat}]

    if isinstance(chat, dict):
        chat = [chat]

    if not isinstance(chat, list):
        raise ValueError("chat must be a list of dictionaries")

    for c in chat:
        if not isinstance(c, dict):
            raise ValueError("chat must be a list of dictionaries")

        if "role" not in c:
            raise ValueError("chat must have a 'role' key")

        if "content" not in c:
            raise ValueError("chat must have a 'content' key")

    # Format the chat using the model's template if available
    try:
        val = tokenizer.apply_chat_template(chat, tokenize=False)
    except:
        val = _manual_chat_format(chat)

    # Return in LLMBoost expected format
    return {
        "id": random.randint(0, int(1e9)),
        "val": val,
    }


def load_flickr30k_dataset(csv_path, image_dir, num_prompts, transform=None):
    """
    Loads the entire Flickr30k dataset into memory as a list of dictionaries.

    Args:
        csv_path (str): Path to the CSV file with annotations.
        image_dir (str): Directory containing the images.
        transform (callable, optional): Optional transform to be applied to each image.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - "image": The loaded image (PIL.Image or transformed image).
              - "path": Path to the image file.
              - "output": Sample descriptions (originally called metadata) for the image from the CSV file.
    """
    # Read the CSV
    annotations = pd.read_csv(csv_path)

    # Initialize an empty dataset array
    dataset = []
    count = 0

    # Check if the number of prompts is greater than the number of images
    if num_prompts > len(annotations):
        raise ValueError(
            f"Number of prompts ({num_prompts}) is greater than the number of images ({len(annotations)})"
        )

    # Iterate through each row in the CSV
    for _, row in tqdm(
        annotations.iterrows(), total=len(annotations), desc="Loading dataset"
    ):
        img_name = row["filename"]
        img_path = os.path.join(image_dir, img_name)

        # Load the image
        try:
            image = Image.open(img_path).convert("RGB")
            if transform:
                image = transform(image)  # Apply any provided transformation
        except FileNotFoundError:
            log.info(f"Warning: Image file {img_path} not found. Skipping...")
            continue

        # Append the image and output as a dictionary
        dataset.append(
            {
                "image": image,
                "path": [img_path],
                "output": row.to_dict(),  # Convert the entire row to a dictionary
                "question": "What is the content of this image?",
            }
        )

        count += 1
        if count >= num_prompts:
            break

    # structure of the dataset
    # array of { "image": PIL.Image, "output": dict }
    return dataset


def merge_image_prompt(raw_prompts: List, target_output: List) -> Tuple[List, List]:
    # Given an **image** dataset, merge several images (according to FLAGS.max_image_per_prompt) into one prompt.

    if FLAGS.query_type != "image":
        raise ValueError("Only support merging prompt with image input")

    if FLAGS.dataset == "flickr30k":
        raise NotImplementedError(
            "Currently doesn't support merging multiple flickr30k images into one prompt"
        )

    ## ensure the length match
    if len(raw_prompts) != len(target_output):
        raise ValueError(
            f"Length mismatch: raw_prompts ({len(raw_prompts)}) and target_output ({len(target_output)})"
        )

    merged_dataset = []
    merged_target_output = []
    total_dataset = len(raw_prompts)

    # generate integer values from a normal distribution constrained between 1 and 10 with a mean of 4,
    mean_val = (
        4 if FLAGS.max_image_per_prompt == 10 else FLAGS.max_image_per_prompt // 2
    )
    samples = np.random.normal(loc=mean_val, scale=1.5, size=FLAGS.num_prompts)
    arr_n_merge_images = np.clip(
        np.round(samples), 1, FLAGS.max_image_per_prompt
    ).astype(int)

    idx = 0
    while len(merged_dataset) < FLAGS.num_prompts:
        # randomize the idx of the prompts to merge
        merge_size = arr_n_merge_images[idx]
        unique_random_ids = np.random.choice(
            range(0, total_dataset - 1), size=merge_size, replace=False
        )

        # merge the prompts
        merged_prompt = {
            "user": raw_prompts[unique_random_ids[-1]][
                "user"
            ],  # use the text prompt of the last prompt
            "system": raw_prompts[unique_random_ids[-1]]["system"],
            "image_path_list": [
                raw_prompts[i]["image_path_list"] for i in unique_random_ids
            ],  # collect all image paths
        }
        merged_dataset.append(merged_prompt)

        # get the target output of the last prompt
        merged_target_output.append(target_output[unique_random_ids[-1]])

        idx += 1

    if len(merged_dataset) != len(merged_target_output):
        raise ValueError(
            f"Merged dataset and target output length mismatch: "
            f"{len(merged_dataset)} != {len(merged_target_output)}"
        )

    return merged_dataset, merged_target_output


def prepare_dataset():
    image_captions = None
    image_dir = None
    dataset = None
    log.info(f"Loading dataset {FLAGS.dataset} (this may take a while)...")
    if FLAGS.dataset == "cnn_dailymail":
        dataset_raw = load_dataset("cnn_dailymail", "3.0.0", split="test")
        # cut down to FLAGS.num_prompts
        dataset_raw = dataset_raw.select(range(FLAGS.num_prompts))
        # features: ['article', 'highlights', 'id'],

        dataset = dataset_raw.to_pandas()
        dataset["system_prompt"] = (
            "You are a summarization system. Summarize the article below.\n\n"
        )
        dataset.rename(
            columns={"article": "question", "highlights": "output"}, inplace=True
        )
    elif FLAGS.dataset == "squad":
        dataset_raw = load_dataset("squad", split="validation")
        # cut down to FLAGS.num_prompts
        dataset_raw = dataset_raw.select(range(FLAGS.num_prompts))
        # features: ['id', 'title', 'context', 'question', 'answers'],

        dataset = dataset_raw.to_pandas()
        dataset["system_prompt"] = (
            "You are a question answering system. Answer the question based on the context.\n\n"
        )
        # merge the context and question into a single string
        dataset["question"] = (
            "Context: "
            + dataset["context"]
            + "\n\n"
            + "Question: "
            + dataset["question"]
        )
        dataset.rename(columns={"answers": "output"}, inplace=True)

    elif FLAGS.dataset == "llava-instruct-mix-vsft":
        if FLAGS.query_type != "image":
            raise ValueError("llava-instruct-mix-vsft dataset is only for image input")
        if FLAGS.dataset_path is None:
            log.error("ERROR: Please specify the dataset_path using --dataset_path")
            exit(1)
        dataset_path = FLAGS.dataset_path
        prompt_path = os.path.join(dataset_path, "extracted", "prompts.csv")
        dataset = pd.read_csv(
            prompt_path
        )  # "path", "question", "output" are the columns
        # prepend the dataset path to the image paths
        dataset["path"] = FLAGS.dataset_path + "/" + dataset["path"]
        log.info(f"Loaded {len(dataset)} samples from {prompt_path}")

    elif FLAGS.dataset == "flickr30k":
        if FLAGS.query_type != "image":
            raise ValueError("flickr30k dataset is only for image input")
        if FLAGS.dataset_path is None:
            log.error("ERROR: Please specify the dataset_path using --dataset_path")
            exit(1)
        dataset_path = FLAGS.dataset_path
        image_captions = os.path.join(dataset_path, "flickr_annotations_30k.csv")
        image_dir = os.path.join(dataset_path, "flickr30k-images")
        dataset = load_flickr30k_dataset(image_captions, image_dir, FLAGS.num_prompts)
        dataset = pd.DataFrame(dataset)
        # Check if CSV and image directory exist
        if not os.path.exists(image_captions):
            raise FileNotFoundError(f"CSV file not found at {image_captions}")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found at {image_dir}")
    else:
        # Load pickle file for llama2 (open orca) dataset
        with open(dataset_info_map[FLAGS.dataset]["dataset_path"], "rb") as f:
            dataset = pickle.load(f)

    # Prepare Prompts
    prompts = []
    n_prompts = 0
    for i, row in dataset.iterrows():
        system = row.get("system_prompt", None)
        user = row.get("question", None)
        image_path_list = row.get("path", [])
        prompts.append(
            {"system": system, "user": user, "image_path_list": image_path_list}
        )
        n_prompts += 1

        # stop when we have enough prompts
        if n_prompts == FLAGS.num_prompts:
            break

    target_texts = dataset[dataset_info_map[FLAGS.dataset]["output_key"]].tolist()
    if len(target_texts) < FLAGS.num_prompts:
        raise ValueError(
            f"length of target_texts ({len(target_texts)}) is smaller than the --num_prompts you speficied ({FLAGS.num_prompts})"
        )
    target_texts = target_texts[: FLAGS.num_prompts]

    if FLAGS.query_type == "image" and FLAGS.max_image_per_prompt > 1:
        # For testing with multiple images per prompt
        prompts, target_texts = merge_image_prompt(prompts, target_texts)

    log.info(f"Loaded dataset with {len(prompts)} samples")
    return prompts, target_texts


def postprocess_text(
    preds: List[str],
    targets: List[str],
) -> Tuple[List[str], List[str]]:
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rogueLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


# General benchmarking code for OpenAI endpoints
async def run_inference_openai(inputs: List[Dict[str, Union[str, int]]]) -> dict:
    pbar = tqdm(total=len(inputs), desc=f"universal_benchmark.py", unit="req")
    _issue_time, _first_token_time, _end_token_time = {}, {}, {}
    preds = {}
    start, end = -1, -1
    lock = threading.Lock()

    if "/v1/chat/completions" in FLAGS.endpoint:
        use_completions = False
    elif "/v1/completions" in FLAGS.endpoint:
        use_completions = True
    else:
        raise ValueError("Invalid endpoint format specified.")

    # NOTE: HTTP1 benchmarking concurrency is purely set by max_parallel_reqs
    max_workers = (
        FLAGS.max_parallel_reqs if FLAGS.max_parallel_reqs > 0 else len(inputs)
    )

    # Single connection, single request
    async def _send(idx: int, item: Dict[str, Union[str, int]], session, lock, pbar):
        # Optional naive rate limiting
        if FLAGS.pps > 0:
            await asyncio.sleep(idx / float(FLAGS.pps))

        # Formulate the endpoint for either an /v1/chat/completions endpoint or an /v1/completions endpoint
        if use_completions:
            payload = {
                "model": FLAGS.model_name,
                "prompt": item["val"],
                "stream": FLAGS.streaming,
            }
        else:
            payload = {
                "model": FLAGS.model_name,
                "messages": [{"role": "user", "content": item["val"]}],
                "stream": FLAGS.streaming,
            }

        t0 = time.perf_counter()
        RETRY_LIMIT = 3
        # placeholders
        first_token = None
        content_accum: List[str] = []

        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                async with session.post(
                    FLAGS.endpoint,
                    json=payload,
                    timeout=FLAGS.timeout,
                    ssl=False,
                ) as response:
                    response.raise_for_status()

                    if not FLAGS.streaming:
                        result = await response.json()
                        if use_completions:
                            content = result["choices"][0]["message"]
                        else:
                            content = result["choices"][0]["text"]
                    else:
                        async for raw_chunk in response.content:
                            line = raw_chunk.decode("utf-8").strip()
                            if not line.startswith("data:"):
                                continue

                            payload_line = line.removeprefix("data:").strip()
                            if payload_line == "[DONE]":
                                break

                            obj = json.loads(payload_line)
                            if use_completions:
                                delta = obj["choices"][0].get("text", "")
                            else:
                                delta = obj["choices"][0]["delta"].get("content", "")

                            if delta:
                                # first token arrival
                                if first_token is None:
                                    first_token = time.perf_counter()
                                content_accum.append(delta)

                        # Accumulate the content into a single string
                        content = "".join(content_accum)

                    if not content:
                        raise ValueError("Empty response content")
                    break  # Success, exit retry loop

            except Exception as e:
                if attempt == RETRY_LIMIT:
                    # explicitly report the error after retries
                    raise ValueError(
                        f"ERROR request {idx}... after {RETRY_LIMIT} retry attempts"
                    )
                else:
                    await asyncio.sleep(1)  # Optional backoff before retrying

        t1 = time.perf_counter()

        async with lock:
            preds[idx] = content
            _issue_time[idx] = t0
            _first_token_time[idx] = (
                first_token or t1
            )  # nonstreaming requests return all the tokens at once
            _end_token_time[idx] = t1
            pbar.update(1)

    # Send the requests
    start = time.perf_counter()

    connector = aiohttp.TCPConnector(limit=max_workers)
    async with aiohttp.ClientSession(connector=connector) as session:
        lock = asyncio.Lock()
        tasks = [
            _send(idx, item, session, lock, pbar) for idx, item in enumerate(inputs)
        ]
        await asyncio.gather(*tasks)

    end = time.perf_counter()
    pbar.close()

    return {
        "_issue_time": _issue_time,
        "_first_token_time": _first_token_time,
        "_end_token_time": _end_token_time,
        "start": start,
        "end": end,
        "preds": preds,
    }


def run_benchmark() -> None:
    # Preparing the benchmarking dataset
    prompts, target_texts = prepare_dataset()
    inputs = []
    log.info(f"FLAGS.model_name: {FLAGS.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_name, trust_remote_code=True)
    for i, p in enumerate(prompts[: FLAGS.num_prompts]):
        if FLAGS.query_type == "image":
            a_prompt_val = [
                {
                    "type": "text",
                    "text": p["user"],
                }
            ]
            for image_path in p["image_path_list"]:
                a_prompt_val.append(
                    {
                        "type": "image",
                        "image": image_path,
                    }
                )
            inputs.append(
                {
                    "id": i,
                    "val": a_prompt_val,
                }
            )
        else:
            a_prompt = [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ]
            inputs.append(
                apply_format(
                    tokenizer,
                    a_prompt,
                )
            )
        inputs[-1]["id"] = i

    # benchmark based on the universal OPENAI API
    # result_dict = run_inference_openai(inputs)
    result_dict = asyncio.run(run_inference_openai(inputs))

    _issue_time = result_dict["_issue_time"]
    _first_token_time = result_dict["_first_token_time"]
    _end_token_time = result_dict["_end_token_time"]
    start = result_dict["start"]
    end = result_dict["end"]
    preds = result_dict["preds"]

    elapsed_time = None
    tokens_per_sec = None
    req_per_sec = None
    prompt_tokens_per_sec = None
    generation_tokens_per_sec = None
    if FLAGS.query_type == "image":
        total_input_len = 0
        total_output_len = 0
        for i in range(len(inputs)):
            total_input_len += len(inputs[i]["val"][0]["text"])
            assert (
                len(preds[i]) > 0
            ), f"Prediction {i} is empty. \n This is the detailed request: {inputs[i]}"
            total_output_len += len(preds[i])
            # calculate the avg for easy metrics calculation later
            input_len = total_input_len // FLAGS.num_prompts
            output_len = total_output_len // FLAGS.num_prompts

        total_num_tokens = FLAGS.num_prompts * (output_len + input_len)
        elapsed_time = end - start
        req_per_sec = FLAGS.num_prompts / elapsed_time
        tokens_per_sec = total_num_tokens / elapsed_time
        prompt_tokens_per_sec = (FLAGS.num_prompts * input_len) / elapsed_time
        generation_tokens_per_sec = (FLAGS.num_prompts * output_len) / elapsed_time

        if FLAGS.snippet:

            # choose 5 random samples
            for _ in range(5):
                i = random.randint(0, len(preds) - 1)
                # Use that number to index into the predictions and target_texts
                log.info(f"Snippet of prediction {i}")
                log.info(f"Prompt:\n {inputs[i]['val']}")
                log.info(f"Prediction:\n {preds[i]}")

    else:
        preds = [preds[key] for key in sorted(preds.keys(), reverse=False)]
        inps = [inp["val"] for inp in inputs]
        inp_ids = tokenizer(inps)["input_ids"]
        preds_ids = tokenizer(preds)["input_ids"]

        if FLAGS.snippet:

            # choose 5 random samples
            for _ in range(5):
                i = random.randint(0, len(preds) - 1)
                # Use that number to index into the predictions and target_texts
                log.info(f"Snippet of prediction {i}")
                log.info(f"Prompt:\n {prompts[i]}")
                log.info(f"Prediction:\n {preds[i]}")
                log.info(f"Target:\n {target_texts[i]}")

        total_input_tokens = 0
        for inp in inp_ids:
            total_input_tokens += len(inp)

        total_output_tokens = 0
        for out in preds_ids:
            total_output_tokens += len(out)

        elapsed_time = end - start
        total_num_tokens = total_input_tokens + total_output_tokens
        req_per_sec = FLAGS.num_prompts / elapsed_time
        tokens_per_sec = total_num_tokens / elapsed_time
        prompt_tokens_per_sec = total_input_tokens / elapsed_time
        generation_tokens_per_sec = total_output_tokens / elapsed_time

    log.info(f"Total time: {elapsed_time} seconds")
    log.info(f"Throughput: {tokens_per_sec:.2f} tokens/s")
    log.info(f"            {req_per_sec:.2f} reqs/s")
    log.info(f"Prompt    : {prompt_tokens_per_sec:.2f} tokens/s")
    log.info(f"Generation: {generation_tokens_per_sec:.2f} tokens/s")


def main(argv):
    del argv  # Unused.
    run_benchmark()


if __name__ == "__main__":
    app.run(main)
