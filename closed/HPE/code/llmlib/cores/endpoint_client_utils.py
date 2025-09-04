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
Shared infrastructure for endpoint client managers

This module provides cold-path utilities shared between FastRequestManager
and OpenAIConcurrentRequestManager. Only non-performance-critical code.
"""

import logging
import multiprocessing as mp
from multiprocessing import managers
import queue
import time
from typing import Any, Callable, List, Optional, Tuple

from transformers import AutoTokenizer

# Shared tokenizer cache across all managers in threading mode
_tokenizer_cache = {}


def get_cached_tokenizer(model_name: str, model_revision: str) -> Any:
    """Get or create cached tokenizer instance (cold path only)"""
    tokenizer_key = (model_name, model_revision)
    if tokenizer_key not in _tokenizer_cache:
        _tokenizer_cache[tokenizer_key] = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision
        )
    return _tokenizer_cache[tokenizer_key]


class PriorityQueueManager(managers.BaseManager):
    """Manager for shared priority queue"""
    pass


# Singleton manager to avoid restarting it
_priority_queue_manager = None


def get_mp_priority_queue():
    """Returns a multiprocess-safe priority queue."""
    global _priority_queue_manager
    if _priority_queue_manager is None:
        PriorityQueueManager.register('PriorityQueue', queue.PriorityQueue)
        _priority_queue_manager = PriorityQueueManager()
        _priority_queue_manager.start()
    return _priority_queue_manager.PriorityQueue()


class WorkerProcessManager:
    """Manages worker process lifecycle for multiprocess mode"""
    
    @staticmethod
    def start_worker_processes(
        worker_count: int,
        worker_main_func: Callable,
        init_args: List[Any],
        process_name_prefix: str = "Worker"
    ) -> List[mp.Process]:
        """
        Start worker processes and wait for them to be ready.

        Returns:
            List of worker processes.
        """
        readiness_queue = mp.Queue()
        worker_processes = []
        
        for i in range(worker_count):
            process = mp.Process(
                target=worker_main_func,
                args=(*init_args, i, readiness_queue),
                name=f"{process_name_prefix}-{i}"
            )
            process.start()
            worker_processes.append(process)
        
        # Wait for all workers to signal readiness
        ready_workers = 0
        start_time = time.time()
        
        try:
            while ready_workers < worker_count:
                message = readiness_queue.get(timeout=30.0)
                if message is True:
                    ready_workers += 1
                else:
                    # Worker reported an error
                    raise RuntimeError(f"Worker initialization failed: {message}")
        except (queue.Empty, RuntimeError) as e:
            # On failure, terminate all started processes
            logging.error(f"Worker startup failed: {e}. Terminating workers.")
            for p in worker_processes:
                if p.is_alive():
                    p.terminate()
                p.join()
            # Re-raise the exception
            if isinstance(e, queue.Empty):
                raise TimeoutError("Worker initialization timed out after 30s") from e
            raise e
        
        elapsed = time.time() - start_time
        logging.info(f"All {worker_count} worker processes ready in {elapsed:.2f}s")
        
        return worker_processes
    
    @staticmethod
    def shutdown_workers(
        worker_processes: List[mp.Process],
        request_queues: List[mp.Queue]
    ) -> None:
        """Shutdown worker processes gracefully"""
        # Signal all workers to stop
        for queue in request_queues:
            queue.put(None)
        
        # Wait for processes to finish
        for process in worker_processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join()


def setup_logging_for_worker():
    """Common logging setup for worker processes"""
    logging.getLogger('openai').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING) 