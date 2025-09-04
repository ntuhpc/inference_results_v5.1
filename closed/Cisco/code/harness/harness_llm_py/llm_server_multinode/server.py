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

from __future__ import annotations
from collections import defaultdict
import os
from pprint import pformat
import signal
from typing import Any, Callable, Dict, List, Tuple
import time
import psutil

from code.common.utils import add_nvtx_scope_wrap, parse_cli_flags

from .config import EngineConfig, HarnessConfig
from .core_multinode import LLMCore
from .utils import LLMServerProgressDisplay, add_prefix_logger
import threading

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

group_size = None
group_id = None
participants = None
device_ids = None

print(f"rank: {rank}, group_id: {group_id}, participants: {participants}")

TAG_BATCH = 9100
TAG_FLUSH = 9101
TAG_COMPLETE = 9102
TAG_FLUSH_ACK = 9104
TAG_STATS= 9105

@add_prefix_logger()
@add_nvtx_scope_wrap()
class LLMServer():
    def __init__(self,
                 scenario: str,
                 devices: List[int],
                 enable_sort: bool,
                 trtllm_checkpoint_flags: Dict[str, Any],
                 trtllm_build_flags: Dict[str, Any],
                 trtllm_runtime_flags: Dict[str, Any],
                 gpu_engine_dir: os.PathLike,
                 gpu_batch_size: int,
                 verbose: bool,
                 verbose_nvtx: bool,
                 log_dir: str,
                 use_graphs: bool,
                 llm_gen_config_path: os.PathLike):
        self.setup_interrupt_handler()
        self.devices = devices
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx

        TP = int(trtllm_build_flags['tensor_parallelism'])
        PP = int(trtllm_build_flags['pipeline_parallelism'])
        
        global group_size, group_id, participants, device_ids

        group_size = PP * TP
        num_cores = (size-1) // group_size
        num_groups = (size-1) // num_cores
        group_id = rank // group_size
        participants = [i for i in range(size) if i // num_groups == group_id and i != (size-1)]
        print(participants)

        device_ids = [id % 8 for id in participants]
        print(f"rank: {rank}, group_id: {group_id}, participants: {participants}")

        # NOTE(vir):
        # we ignore trtllm_build_flags['max_batch_size'] if given,
        # use legacy field gpu_batch_size instead
        if trtllm_build_flags.get('max_batch_size', -1) != gpu_batch_size:
            trtllm_build_flags['max_batch_size'] = gpu_batch_size
            self.logger.info(f'Overriding trtllm_build_flags[max_batch_size] with legacy field gpu_batch_size={gpu_batch_size}')

        if 'max_batch_size' not in trtllm_runtime_flags:
            trtllm_runtime_flags['max_batch_size'] = gpu_batch_size
            self.logger.info(f'Using engine max-batch-size as runtime max-batch-size since its not specified in trtllm_runtime_flags')

        if 'max_num_tokens' not in trtllm_runtime_flags:
            trtllm_runtime_flags['max_num_tokens'] = trtllm_build_flags['max_num_tokens']
            self.logger.info(f'Using engine max-num-tokens as runtime max-num-tokens since its not specified in trtllm_runtime_flags')

        # set remaining defaults
        trtllm_runtime_flags |= {
            'batch_scheduler_policy': trtllm_runtime_flags.get('batch_scheduler_policy', 'max_util'),
            'context_chunking_policy': trtllm_runtime_flags.get('context_chunking_policy', 'first_come_first_served'),
            'use_inflight_batching': trtllm_runtime_flags.get('use_inflight_batching', True),
            'enable_batch_size_tuning': trtllm_runtime_flags.get('enable_batch_size_tuning', False),
            'enable_max_num_tokens_tuning': trtllm_runtime_flags.get('enable_max_num_tokens_tuning', False),
            'dynamic_batch_moving_average_window': trtllm_runtime_flags.get('dynamic_batch_moving_average_window', 128),
            'kvcache_free_gpu_mem_frac': trtllm_runtime_flags.get('kvcache_free_gpu_mem_frac', 0.80),
            'enable_chunked_context': trtllm_runtime_flags.get('enable_chunked_context', False),
            'exclude_input_from_output': trtllm_runtime_flags.get('exclude_input_from_output', True)
        }

        # override runtime flags
        if runtime_overrides := os.environ.get('TRTLLM_RUNTIME_FLAGS', None):
            self.logger.info(f"Detected TRTLLM_RUNTIME_FLAGS: {runtime_overrides}")
            runtime_overrides = parse_cli_flags(runtime_overrides)
            for key, override in runtime_overrides.items():
                self.logger.info(f"Overriding {key}: {override}")
                trtllm_runtime_flags[key] = override

        self.engine_config = EngineConfig.from_engine_dir(gpu_engine_dir)
        self.harness_config = HarnessConfig(
            traffic_distribution_policy="load_balancing" if scenario != "Offline" else "round_robin",
            gen_config=HarnessConfig.load_generation_config(llm_gen_config_path),
            trtllm_checkpoint_flags=trtllm_checkpoint_flags,
            trtllm_build_flags=trtllm_build_flags,
            trtllm_runtime_flags=trtllm_runtime_flags,
            log_dir=log_dir,
        )
        self.harness_config.gen_config.streaming &= scenario != 'Offline'
        self.harness_config.validate_compatible_engine(self.engine_config)

        self.logger.info(f'HarnessConfig: \n{pformat(self.harness_config, compact=True)}')
        self.logger.info(f'EngineConfig: \n{pformat(self.engine_config, compact=True)}')

        print(f"rank {rank} starting LLMServer")
        self.start()

    def __del__(self):
        self.logger.verbose("Destructor invoked.")
        if hasattr(self, 'core'):
            if self.core:
                self.core.notify_stop()
                del self.core

        # stop any pending samples thread
        if hasattr(self, '_pending_samples_thread'):
            self._pending_samples_thread.join()
            del self._pending_samples_thread
        
        # stop any completion thread
        if hasattr(self, '_completion_thread'):
            self._shutdown = True
            self._completion_thread.join()
            del self._completion_thread
        
        self.logger.verbose("Destructor completed.")

    def setup_interrupt_handler(self):
        current_process = psutil.Process()

        def exit_fn(signum, frame):
            self.logger.info("Received SIGINT. Stop LLMServer and cleanup.")

            children = current_process.children(recursive=True)
            for child in children:
                self.logger.verbose(f"Sending SIGKILL to child process: {child.pid}")
                os.kill(child.pid, signal.SIGKILL)

        signal.signal(signal.SIGINT, exit_fn)

    def start(self):
        self.logger.verbose("start() invoked.")

        # reset state
        self.sample_count = 0

        print(f"rank {rank} initializing Core")
        self.core: LLMCore = None
        self.initialize_core()

        if rank % group_size == 0 and rank != size - 1:
            self.start_batch_receiver()

        comm.Barrier()

        self.logger.verbose("start() completed.")

    def initialize_core(self):
        """
        Initialize LLMCore instances.
        """

        # init core
        self.core = LLMCore(
                name=f'core#{rank}',
                device_ids=device_ids,
                participant_ids=participants,
                complete_callback=self.complete_request,
                send_stats_callback=self.send_stats,
                engine_config=self.engine_config,
                harness_config=self.harness_config,
                verbose=self.verbose,
                verbose_nvtx=self.verbose_nvtx,
            )
    
    def stop_work(self):
        """
        Stop accepting new requests and signal LLMCores to complete all pending work.
        Cleanup corresponding Issue and Response resources.
        """
        
        self._shutdown = True

        self.logger.verbose(f"stop_work() invoked.")
        with self.nvtx_scope("stop_work"):
            # signal LLMCores to exit
            self.core.notify_stop()

            print(f"stopping completion thread on rank {rank}")
            if hasattr(self, "_completion_thread"):
                self._completion_thread.join()

            # cleanup gpu resources
            print(f"rank {rank} cleaning up core")
            del self.core

        self.logger.info(f"Total Samples Completed={self.sample_count}")
        self.logger.verbose(f"stop_work() completed.")

    # needs update to support multiple ranks
    def warm_up(self):
        """
        Run Warm up iterations on all LLMCores.
        """
        with self.nvtx_scope("warm_up"):
            self.core.warm_up(warm_up_iters=100)

    def start_batch_receiver(self):
        if rank % group_size != 0:
            return

        def _rx():
            print(f"rank {rank} starting batch receiver thread")
            while not self.core.stop_work.is_set():
                if comm.Iprobe(source=(size-1), tag=TAG_BATCH):
                    batch = comm.recv(source=(size-1), tag=TAG_BATCH)
                    self.core.enqueue(batch)

                elif comm.Iprobe(source=size-1, tag=TAG_FLUSH):
                    _ = comm.recv(source=size-1, tag=TAG_FLUSH)
                    with self.nvtx_scope("flush_core"):
                        self.core.flush()

                    comm.isend(None, dest=size-1, tag=TAG_FLUSH_ACK).Wait()
                    self.core.notify_stop()
                    break

                else:
                    time.sleep(0.001)
                
        threading.Thread(target=_rx, daemon=True).start()

    def issue_queries(self, query_samples: List[Tuple[int, List[int], List[int]]]):
        """Distribute query_samples across all TP-leader ranks.""" 
        pass

    @staticmethod
    def complete_request(request_id: int, output_tokens: List[int], is_first_token: bool):
        comm.isend((request_id, output_tokens, is_first_token), dest=size-1, tag=TAG_COMPLETE).Wait()
    
    @staticmethod
    def send_stats(additional_unit_updates):
        core_num = rank // group_size
        comm.isend((additional_unit_updates, core_num), dest=size-1, tag=TAG_STATS).Wait()
