# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""
Fast HTTP Client for TensorRT-LLM Endpoint
"""

import asyncio
import collections
from dataclasses import dataclass, field
import datetime
import logging
import multiprocessing as mp
import os
import queue
import signal
from typing import Any, Dict, List, Optional, Tuple

import httpx
import orjson
from transformers import AutoTokenizer
import uvloop

from ..config import HarnessWorkerMode, TrtllmEndpointConfig
from .base import LLMRequest, LLMResponse
from . import endpoint_client_utils

# lazy init for global module loop
_module_loop = None


def _get_module_loop():
    """Get the unified module loop with lazy import"""
    global _module_loop
    if _module_loop is None:
        from . import trtllm_endpoint_core
        _module_loop = trtllm_endpoint_core._module_loop
    return _module_loop


# Shared tokenizer cache for threading mode
_tokenizer_cache = {}


class RetryableException(Exception):
    """Custom exception for retryable errors"""
    pass


@dataclass(slots=True, order=True)
class FastRequest:
    """Optimized request structure with slots for memory efficiency"""
    priority: int
    request_id: int = field(compare=False)
    input_tokens: List[int] = field(compare=False)
    stop_tokens: Optional[List[int]] = field(compare=False, default=None)
    retry_count: int = field(compare=False, default=0)


class FastRequestManager:
    """Fast request manager with lock-free operations supporting both threading and multiprocessing modes"""

    def __init__(
        self,
        config: TrtllmEndpointConfig,
        max_concurrency: int,
        workers_per_core: int,
        mode: HarnessWorkerMode,
    ):
        self.config = config
        self.max_concurrency = max_concurrency
        self.workers_per_core = workers_per_core
        self.mode = mode
        self.max_retries = 10  # Max retries for a request

        # Extract model info
        self.model_name, self.model_revision = list(config.model_repo.items())[0]

        self._response_queue = None  # Will be initialized based on mode
        self._initialize()

    def _initialize(self):
        """Initialize based on execution mode"""
        if self.mode == HarnessWorkerMode.MULTIPROCESS:
            self._init_multiprocess()
        else:
            self._init_threading()

    def _init_threading(self):
        """Initialize threading mode with shared HTTP client and priority queue."""
        self._response_queue = queue.Queue(maxsize=10000)
        self._request_queue = asyncio.PriorityQueue()

        # Use shared tokenizer cache for threading mode
        tokenizer_key = (self.model_name, self.model_revision)
        if tokenizer_key not in _tokenizer_cache:
            _tokenizer_cache[tokenizer_key] = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.model_revision
            )
        self.tokenizer = _tokenizer_cache[tokenizer_key]

        # Use module-level event loop
        self.loop = _get_module_loop()

        # Create fast HTTP client
        self.http_client = FastHTTPClient(
            endpoint_url=self.config.endpoint_url,
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            config=self.config,
            max_concurrency=self.max_concurrency,
            response_callback=self._handle_response,
        )

        # Start dispatcher task
        self._dispatcher_task = asyncio.run_coroutine_threadsafe(
            self._dispatcher_loop(), self.loop
        )

    def _init_multiprocess(self):
        """Initialize multiprocess mode with worker processes and a shared priority queue."""
        self._response_queue = mp.Queue(maxsize=10000)
        self._request_queue = endpoint_client_utils.get_mp_priority_queue()

        # Distribute concurrency among workers
        concurrency_per_worker = self.max_concurrency

        # Start worker processes using the utility
        self.worker_processes = endpoint_client_utils.WorkerProcessManager.start_worker_processes(
            worker_count=self.workers_per_core,
            worker_main_func=FastRequestManager._worker_process_main,
            init_args=(
                self.config,
                concurrency_per_worker,
                self._request_queue,
                self._response_queue,
            ),
            process_name_prefix="FastHTTPWorker"
        )

    async def _dispatcher_loop(self):
        """Main loop for dispatching requests in threading mode"""
        while True:
            # Get highest priority request
            request = await self._request_queue.get()

            try:
                await self.http_client.submit_request(request)
            except RetryableException:
                request.retry_count += 1
                if request.retry_count < self.max_retries:
                    request.priority = -request.retry_count
                    await self._request_queue.put(request)
                else:
                    err_msg = f"Request failed after {self.max_retries} attempts"
                    self._handle_response(request.request_id, None, True, err_msg)
            except Exception as e:
                # Handle non-retryable exceptions from submit_request
                err_msg = f"Non-retryable error during request dispatch: {e}"
                logging.error(f"Request {request.request_id} failed: {err_msg}")
                self._handle_response(request.request_id, None, True, err_msg)

            self._request_queue.task_done()

    def submit_requests(self, queries: List[LLMRequest]) -> None:
        """Submit requests by adding them to the priority queue."""
        for query in queries:
            request = FastRequest(
                priority=0,
                request_id=query.request_id,
                input_tokens=query.input_tokens,
                stop_tokens=query.stop_tokens
            )

            if self.mode == HarnessWorkerMode.MULTIPROCESS:
                self._request_queue.put(request)
            else:
                # For asyncio, schedule put in the event loop
                self.loop.call_soon_threadsafe(self._request_queue.put_nowait, request)

    def _handle_response(self,
                         request_id: int,
                         content: Optional[str],
                         is_final: bool,
                         error: Optional[str]):
        """Fast response handling"""
        if error:
            response = LLMResponse(
                request_id=request_id,
                output_tokens=[],
                is_final_token=True,
                error=error
            )
        else:
            # Convert content to tokens
            tokens = self.tokenizer.encode(content)
            output_tokens = [tokens] if tokens else [[]]

            response = LLMResponse(
                request_id=request_id,
                output_tokens=output_tokens,
                is_final_token=is_final,
                error=None
            )

        # Put in response queue
        self._response_queue.put(response)

    def get_responses(self, timeout: datetime.timedelta) -> List[LLMResponse]:
        """Fast batch response retrieval"""
        timeout_seconds = timeout.total_seconds()
        responses = []

        # Get responses with timeout
        if timeout_seconds > 0:
            try:
                response = self._response_queue.get(timeout=timeout_seconds)
                responses.append(response)

                # Get any additional available responses
                while True:
                    try:
                        response = self._response_queue.get_nowait()
                        responses.append(response)
                    except queue.Empty:
                        break
            except queue.Empty:
                pass  # No responses available within timeout
        else:
            try:
                response = self._response_queue.get_nowait()
                responses.append(response)
            except queue.Empty:
                pass  # No responses available

        return responses

    def shutdown(self):
        """Clean shutdown based on mode"""
        if self.mode == HarnessWorkerMode.MULTIPROCESS:
            logging.debug("Shutting down FastRequestManager multiprocess workers...")
            # Signal all workers to stop by putting None in the shared queue
            for _ in range(self.workers_per_core):
                self._request_queue.put(None)

            # Use utility for shutdown
            endpoint_client_utils.WorkerProcessManager.shutdown_workers(
                self.worker_processes,
                []  # No individual queues to signal
            )
            logging.debug("FastRequestManager multiprocess workers shutdown complete")
        else:
            # For threading mode, clean up HTTP client
            if self._dispatcher_task:
                self._dispatcher_task.cancel()
            asyncio.run_coroutine_threadsafe(self.http_client.shutdown(), self.loop).result(timeout=2.0)
            logging.debug("FastRequestManager HTTP client shutdown complete")

    @classmethod
    def _worker_process_main(cls,
                             config: TrtllmEndpointConfig,
                             max_concurrency: int,
                             request_queue: queue.PriorityQueue,
                             response_queue: mp.Queue,
                             worker_id: int,
                             readiness_queue: mp.Queue):
        """Main function for multiprocess worker. The readiness_queue is used by WorkerProcessManager."""
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logging.info(f"Worker {worker_id} received signal {signum}, shutting down...")
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Set up logging for worker process
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)

        # Disable tokenizer parallelism in worker process
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        try:
            # Create new event loop for this process
            loop = uvloop.new_event_loop()
            asyncio.set_event_loop(loop)

            # All setup is complete, create the worker loop task
            worker_loop_task = loop.create_task(cls._worker_loop(
                config,
                max_concurrency,
                worker_id,
                request_queue,
                response_queue
            ))

            # Signal that this worker is ready to the main process
            readiness_queue.put(True)

            # Run the worker loop until it completes
            loop.run_until_complete(worker_loop_task)
            loop.close()
        except Exception as e:
            # Report failure to the main process
            logging.error(f"Worker {worker_id} failed during setup: {e}")
            readiness_queue.put(f"Worker {worker_id} failed: {e}")
            exit(1)

    @classmethod
    async def _worker_loop(
        cls,
        config: TrtllmEndpointConfig,
        max_concurrency: int,
        worker_id: int,
        request_queue: queue.PriorityQueue,
        response_queue: mp.Queue,
    ):
        """Worker loop for multiprocess mode"""
        # Load tokenizer for this worker
        model_name, model_revision = list(config.model_repo.items())[0]
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
        max_retries = 10

        # Set up response callback that puts responses in the queue
        def response_callback(
            request_id: int,
            content: Optional[str],
            is_final: bool,
            error: Optional[str],
        ):
            if error:
                response = LLMResponse(
                    request_id=request_id,
                    output_tokens=[],
                    is_final_token=True,
                    error=error
                )
            else:
                # Tokenize the content chunk
                tokens = tokenizer.encode(content)
                output_tokens = [tokens]

                response = LLMResponse(
                    request_id=request_id,
                    output_tokens=output_tokens,
                    is_final_token=is_final,
                    error=None
                )

            response_queue.put(response)

        # Create HTTP client for this worker
        http_client = FastHTTPClient(
            endpoint_url=config.endpoint_url,
            model_name=model_name,
            tokenizer=tokenizer,
            config=config,
            max_concurrency=max_concurrency,
            response_callback=response_callback
        )

        loop = asyncio.get_event_loop()

        while True:
            try:
                # Blocking get from the shared priority queue
                request = await loop.run_in_executor(None, request_queue.get)
                if request is None:  # Shutdown signal
                    break
            except (queue.Empty, EOFError):
                # Queue is empty or closed, continue or exit
                await asyncio.sleep(0.01)
                continue

            try:
                await http_client.submit_request(request)
            except RetryableException:
                request.retry_count += 1
                if request.retry_count < max_retries:
                    request.priority = -request.retry_count
                    request_queue.put(request)
                else:
                    error_msg = f"Request failed after {max_retries} attempts"
                    response_callback(request.request_id, None, True, error_msg)
            except Exception as e:
                # Handle non-retryable exceptions
                error_msg = f"Non-retryable error in worker: {e}"
                logging.error(f"Request {request.request_id} failed: {error_msg}")
                response_callback(request.request_id, None, True, error_msg)

        # Clean up HTTP client
        await http_client.shutdown()


class FastHTTPClient:
    """Fast HTTP client with minimal overhead"""

    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        tokenizer: AutoTokenizer,
        config: TrtllmEndpointConfig,
        response_callback: callable,
        max_concurrency: int,
    ):
        self.endpoint_url = f"http://{endpoint_url}/v1/chat/completions"
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.config = config
        self.max_concurrency = max_concurrency

        # Pre-allocate buffers for zero-copy operations
        self.request_buffer_pool = collections.deque()
        self.response_buffer_pool = collections.deque()
        for _ in range(max_concurrency):
            self.request_buffer_pool.append(bytearray(8192))   # 8KB pre-allocated
            self.response_buffer_pool.append(bytearray(16384))  # 16KB pre-allocated

        # lightweight http client with tight timeouts
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=self.max_concurrency,
            ),
            timeout=httpx.Timeout(None),
            follow_redirects=False,
        )

        # Pre-compile headers
        self.headers = {
            'content-type': 'application/json',
            'authorization': 'Bearer dummy',
            'connection': 'keep-alive',
            'accept': 'text/event-stream' if config.gen_config.streaming else 'application/json',
        }

        # Use native asyncio semaphore for better performance
        self._semaphore = asyncio.Semaphore(max_concurrency)

        # Single callback for all responses
        self.response_callback = response_callback

    def _build_request_bytes(self, request: FastRequest, buffer: bytearray) -> int:
        """Build request directly into pre-allocated buffer"""
        gen_params = {
            "model": list(self.config.model_repo.keys())[0],
            "max_tokens": self.config.gen_config.max_output_len,
            "temperature": self.config.gen_config.temperature,
            "top_p": self.config.gen_config.top_p,
            "stream": self.config.gen_config.streaming,
            "messages": [],
            "prompt_token_ids": request.input_tokens,
            "min_tokens": self.config.gen_config.min_output_len,
            "top_k": self.config.gen_config.top_k,
        }

        # Add stop tokens if configured
        if self.config.gen_config.use_stop_tokens and request.stop_tokens:
            gen_params["stop_token_ids"] = request.stop_tokens

        # Serialize directly into buffer using orjson
        data = orjson.dumps(gen_params)
        data_len = len(data)
        buffer[:data_len] = data

        return data_len

    def _should_retry(self, e: Exception) -> bool:
        """Check if error should be retried based on OpenAI's retry patterns"""
        if isinstance(e, httpx.HTTPStatusError):
            # Retry on specific status codes
            status = e.response.status_code
            return status in (408, 409, 429) or status >= 500
        elif isinstance(e, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)):
            # Always retry network errors
            return True
        return False

    async def submit_request(self, request: FastRequest):
        """Submit request and raise RetryableException on failure."""
        async with self._semaphore:
            request_buffer = self.request_buffer_pool.popleft() if self.request_buffer_pool else bytearray(8192)

            try:
                data_len = self._build_request_bytes(request, request_buffer)

                try:
                    response = await self.client.post(
                        self.endpoint_url,
                        content=bytes(request_buffer[:data_len]),
                        headers=self.headers,
                    )

                    if self.config.gen_config.streaming:
                        await self._process_sse_stream(response, request.request_id)
                    else:
                        await self._process_non_streaming(response, request.request_id)

                except Exception as e:
                    if self._should_retry(e):
                        logging.warning(f"Retryable error for request {request.request_id}: {type(e).__name__}: {str(e)}")
                        raise RetryableException from e
                    else:
                        logging.error(f"Non-retryable error for request {request.request_id}: {type(e).__name__}: {str(e)}")
                        self.response_callback(request.request_id, None, True, f"Request failed: {str(e)}")

            finally:
                if len(self.request_buffer_pool) < self.max_concurrency:
                    self.request_buffer_pool.append(request_buffer)

    async def _process_non_streaming(self, response, request_id: int):
        """Process non-streaming response with minimal overhead"""
        response.raise_for_status()
        response_bytes = await response.aread()
        data = orjson.loads(response_bytes)
        response_text = data['choices'][0]['message']['content']
        self.response_callback(request_id, response_text, True, None)

    async def _process_sse_stream(self, response, request_id: int):
        """Fast SSE stream processing with zero allocations"""
        response_buffer = self.response_buffer_pool.popleft() if self.response_buffer_pool else bytearray(16384)
        output_text = ""
        first_chunk_processed = False

        try:
            response.raise_for_status()
            buffer_pos = 0
            line_start = 0

            async for chunk in response.aiter_bytes(chunk_size=8192):
                chunk_len = len(chunk)
                if buffer_pos + chunk_len > len(response_buffer):
                    new_size = max(buffer_pos + chunk_len, len(response_buffer) * 2)
                    new_buffer = bytearray(new_size)
                    new_buffer[:buffer_pos] = response_buffer[:buffer_pos]
                    response_buffer = new_buffer

                response_buffer[buffer_pos:buffer_pos + chunk_len] = chunk
                buffer_pos += chunk_len

                while True:
                    line_end = response_buffer.find(b'\n', line_start, buffer_pos)
                    if line_end == -1:
                        break

                    if response_buffer[line_start:line_start + 6] == b'data: ':
                        content, is_done = self._parse_sse_line_fast(
                            memoryview(response_buffer)[line_start + 6:line_end],
                            request_id
                        )

                        if content:
                            output_text += content
                            if not first_chunk_processed:
                                self.response_callback(request_id, content, False, None)
                                first_chunk_processed = True

                        if is_done:
                            # Final response with full text
                            self.response_callback(request_id, output_text, True, None)

                            # we free buffer to pool in finally block
                            return

                    line_start = line_end + 1

                # Move remaining data to start if needed
                if line_start > 0 and line_start < buffer_pos:
                    remaining = buffer_pos - line_start
                    response_buffer[:remaining] = response_buffer[line_start:buffer_pos]
                    buffer_pos = remaining
                    line_start = 0
                elif line_start >= buffer_pos:
                    buffer_pos = 0
                    line_start = 0

        finally:
            # Always return buffer to pool
            if len(self.response_buffer_pool) < self.max_concurrency:
                self.response_buffer_pool.append(response_buffer)

    def _parse_sse_line_fast(self, data: memoryview, request_id: int) -> Tuple[Optional[str], bool]:
        """Fast SSE line parsing, returns (content, is_done)"""
        if len(data) >= 6 and data[:6].tobytes() == b'[DONE]':
            return None, True
        if len(data) == 0:
            return None, False

        try:
            parsed = orjson.loads(data)
        except orjson.JSONDecodeError:
            logging.error(f"Failed to decode SSE data for request {request_id}: {bytes(data)}")
            return None, False

        choice = parsed['choices'][0]
        finish_reason = choice.get('finish_reason')
        is_done = finish_reason is not None

        delta = choice['delta']
        content = delta.get('content')

        return content, is_done

    async def shutdown(self):
        """Clean shutdown"""
        await self.client.aclose()
