# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import numpy as np

import threading
import queue
import signal
import sys
import json

import logging
from typing import List

import mlperf_loadgen as lg
from dataset import Dataset

log = logging.getLogger("Llama-8B-SUT") 
sys.path.insert(0, "/tmp/build/MLPerf")
import driveos_llm
from plugin_loader import load_plugins

# Load TensorRT plugins first (similar to C++ loadPlugins())
print("Loading TensorRT plugins...")
plugin_handles = load_plugins(int4_gemm_plugin=True)
print(f"Loaded {len(plugin_handles)} plugins")



def complete_loadgen_request(request_id: int, output_tokens: List[int], is_first_token: bool):
    complete_fn = lg.FirstTokenComplete if is_first_token else lg.QuerySamplesComplete
    output_tokens_t = np.ascontiguousarray(output_tokens, dtype=np.uint32)
    output_seq_len = len(output_tokens_t)
    output_toks_ptr = output_tokens_t.ctypes.data
    output_toks_size = output_seq_len * output_tokens_t.itemsize
    complete_fn([lg.QuerySampleResponse(request_id, output_toks_ptr, output_toks_size, output_seq_len)])




class SUT:
    # Maximum number of output tokens to generate
    MAX_OUTPUT_TOKENS = 128
    
    def __init__(
        self,
        model_path=None,
        batch_size=None,
        total_sample_count=13368,
        dataset_path=None,
        scenario=None,
        token_output_file="tokens_output.json",
        engine_dir="/home/engines/8B_fp8.engine",
        async_mode=False
    ):

        self.model_path = model_path or f"meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.engine_dir = engine_dir

        if not batch_size:
            batch_size = 1
        self.batch_size = batch_size

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )
        self.load_model()
        
        self.worker_thread = None
        self.query_queue = queue.Queue()
        self.scenario = scenario
        self.token_output_file = token_output_file
        
        # Synchronization for proper shutdown coordination
        self.queries_issued = 0
        self.issue_complete = False
        self.sync_lock = threading.Lock()
        self.async_mode = async_mode
        
        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.count = 0
        
        # Create a fresh JSON file at the beginning
        self._initialize_token_file()

    def _initialize_token_file(self):
        """Initialize a fresh JSONL file for token logging"""
        try:
            with open(self.token_output_file, 'w', encoding='utf-8') as f:
                pass  # Create empty file
            log.info(f"Initialized fresh token output file: {self.token_output_file}")
        except Exception as e:
            log.error(f"Failed to initialize token output file: {e}")

    def log_tokens_to_json(self, query_ids, input_ids, output_ids, input_text, output_text):
        """
        Log tokens and text to JSONL file (one JSON object per line).
        
        Args:
            query_ids: Query sample library ID
            input_ids: List of input token IDs
            output_ids: List of output token IDs  
            input_text: Detokenized input text
            output_text: Detokenized output text
        """
        try:
            # Create the data entry
            entry = {
                "query_ids": query_ids,
                "input_text": input_text,
                "output_ids": output_ids,
                "output_text": output_text
            }
            
            # Append as a single line (JSONL format) - much more efficient!
            with open(self.token_output_file, 'a', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')  # Add newline for JSONL format
            
            log.info(f"Logged tokens to JSONL - Query ID: {query_ids}, Output tokens: {len(output_ids)}")
            
        except Exception as e:
            log.error(f"Failed to log tokens to JSONL: {e}")

    def _sort_jsonl_by_qsl_index(self):
        """Sort the JSONL file by qsl_index after all processing is complete"""
        try:
            # Read all entries from the JSONL file
            entries = []
            if os.path.exists(self.token_output_file):
                with open(self.token_output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                entry = json.loads(line)
                                entries.append(entry)
                            except json.JSONDecodeError as e:
                                log.warning(f"Skipping invalid JSON line: {line[:100]}... Error: {e}")
                
                # Sort entries by qsl_index
                entries.sort(key=lambda x: x.get('qsl_index', 0))
                
                # Write sorted entries back to the file
                with open(self.token_output_file, 'w', encoding='utf-8') as f:
                    for entry in entries:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write('\n')
                
                log.info(f"Successfully sorted {len(entries)} entries in {self.token_output_file} by qsl_index")
            else:
                log.warning(f"Token output file {self.token_output_file} does not exist, skipping sort")
                
        except Exception as e:
            log.error(f"Failed to sort JSONL file by qsl_index: {e}")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and other termination signals"""
        log.info(f"Received signal {signum}. Initiating graceful shutdown...")
        # Raise KeyboardInterrupt to be caught by main thread
        raise KeyboardInterrupt("Signal received")

    def start(self):
        # Create worker threads
        if self.scenario == "offline":
            self.worker_thread = threading.Thread(target=self.process_queries_offline)
        elif self.scenario == "singlestream":
            self.worker_thread = threading.Thread(target=self.process_queries_singlestream)
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}")
        self.worker_thread.start()

    def stop(self):
        log.info("Stopping SUT...")
   
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                self.query_queue.put(None)
                log.warning("Worker thread did not stop within timeout")
        self._sort_jsonl_by_qsl_index()

    def process_queries_offline(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        raise NotImplementedError("Offline mode is not implemented")

    def process_queries_singlestream(self):
        while True:
            try:
                qitem = self.query_queue.get(timeout=0.05)
                if qitem is None:
                    log.info("Received shutdown signal (None), breaking immediately")
                    break
            except queue.Empty:
                continue
            assert len(qitem) == 1
            query_ids = qitem[0].index

            input_ids_tensor = self.data_object.input_ids[query_ids]
            if not self.async_mode:
                # sync mode for debugging, not used by submission
                print("start sync generation")
                self.model.start_streaming_generation(input_ids_tensor, self.MAX_OUTPUT_TOKENS)
                tokens = self.model.get_streaming_tokens()
                token_ids = []
                for i in tokens:
                    token_ids.append(i["token_ids"][0])
                
                # Get input text
                input_text = self.data_object.tokenizer.decode(input_ids_tensor)
                # Detokenize the output tokens to get readable text
                output_text = self.data_object.tokenizer.decode(token_ids)
                
                # Log to JSON file
                self.log_tokens_to_json(
                    query_ids=query_ids,
                    input_ids=input_ids_tensor.tolist() if hasattr(input_ids_tensor, 'tolist') else list(input_ids_tensor),
                    output_ids=token_ids,
                    input_text=input_text,
                    output_text=output_text
                )
                print("len of token_ids", len(token_ids))
                complete_loadgen_request(qitem[0].id, token_ids, is_first_token=False)
    
            else:
                # async mode - streaming generation with real-time monitoring
                print("start async generation")
                self.model.start_streaming_generation_async(input_ids_tensor, self.MAX_OUTPUT_TOKENS)
                
                # Track tokens and state
                last_token_count = 0
                all_token_ids = []
                first_token_sent = False
                iteration_count = 0
                max_iterations = 100000  # Prevent infinite loop
                monitoring_interval = 0.001  # 1ms polling interval
                
                # Monitor tokens in real-time
                while not self.model.is_streaming_finished() and iteration_count < max_iterations:
                    tokens = self.model.get_streaming_tokens()
                    iteration_count += 1
                    
                    # Process new tokens
                    if len(tokens) > last_token_count:
                        # Process only new tokens
                        new_tokens = tokens[last_token_count:]
                        
                        for i, token_data in enumerate(new_tokens):
                            token_ids = token_data["token_ids"]
                            is_first = token_data["is_first_token"]
                            is_last = token_data["is_last_token"]
                            
                            if token_ids:
                                # Add to our accumulated tokens
                                all_token_ids.extend(token_ids)
                                
                                # Send first token response if this is the first token and we haven't sent it yet
                                if is_first and not first_token_sent:
                                    complete_loadgen_request(qitem[0].id, token_ids, is_first_token=True)
                                    first_token_sent = True
                        
                        last_token_count = len(tokens)
                    
                    # Sleep for the monitoring interval
                    time.sleep(monitoring_interval)
                
                # Handle case where maximum iterations reached
                if iteration_count >= max_iterations:
                    log.warning(f"Reached maximum iterations ({max_iterations}), generation may have stalled")

                # Process any remaining tokens after generation finished
                final_tokens = self.model.get_streaming_tokens()
                if len(final_tokens) > last_token_count:
                    # Process remaining tokens
                    remaining_tokens = final_tokens[last_token_count:]
                    for token_data in remaining_tokens:
                        token_ids = token_data["token_ids"]
                        if token_ids:
                            all_token_ids.extend(token_ids)
                # Send complete response with all accumulated tokens
                if all_token_ids:
                    # Get input and output text for logging
                    input_text = self.data_object.tokenizer.decode(input_ids_tensor)
                    output_text = self.data_object.tokenizer.decode(all_token_ids)
                    
                    # Log to JSON file
                    # self.log_tokens_to_json(
                    #     query_ids=query_ids,
                    #     input_ids=input_ids_tensor.tolist() if hasattr(input_ids_tensor, 'tolist') else list(input_ids_tensor),
                    #     output_ids=all_token_ids,
                    #     input_text=input_text,
                    #     output_text=output_text
                    # )
                    
                    # Send final complete response
                    complete_loadgen_request(qitem[0].id, all_token_ids, is_first_token=False)
                else:
                    log.warning("No tokens were generated during async streaming")

    def load_model(self):
        log.info("Loading model...")

        # For single GPU setup
        self.model = driveos_llm.LLMChat()
        self.model.load_engine(self.engine_dir, self.model_path)

        log.info("Loaded model")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def issue_queries(self, query_samples):
        """Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        log.info(f"IssueQuery started with {len(query_samples)} samples")
        
        with self.sync_lock:
            self.queries_issued = 0
            self.issue_complete = False
        
        while len(query_samples) > 0:
            batch = query_samples[: self.batch_size]
            self.query_queue.put(batch)
            
            with self.sync_lock:
                self.queries_issued += len(batch)
            
            query_samples = query_samples[self.batch_size:]
            
        log.info(f"IssueQuery done - issued {self.queries_issued} total queries")
        self.count += 1
        with self.sync_lock:
            self.issue_complete = True

    def flush_queries(self):
        log.info("FlushQueries started")
        # Don't put None here - let process_queries finish naturally
        # None is only for final shutdown in stop()



