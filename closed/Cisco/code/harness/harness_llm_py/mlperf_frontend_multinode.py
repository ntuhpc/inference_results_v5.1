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


import inspect
import os
import pickle as pkl
from typing import List
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from code.common import args_to_string, arguments as common_args, logging, run_command
from code.common.harness import BaseBenchmarkHarness
from code.harness.harness_llm_py.llm_server_multinode import LLMServer
from code.harness.harness_llm_py.llm_server import LLMDataset
from .llm_server_multinode.utils import LLMServerProgressDisplay, add_prefix_logger


import threading
import time

from tqdm import tqdm

from mpi4py import MPI

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

TAG_BATCH = 9100
TAG_FLUSH = 9101
TAG_COMPLETE = 9102
TAG_FLUSH_ACK = 9104
TAG_STATS= 9105

_live_buffers = {}

try:
    import mlperf_loadgen as lg
except:
    logging.warning("Loadgen Python bindings are not installed. Installing Loadgen Python bindings!")
    run_command("make build_loadgen")
    import mlperf_loadgen as lg


def create_qsl_cls_mn(dataset_cls: type) -> type:
    """
    Generate a Mlperf-Inference QSL Wrapper for given LLMDataset.
    This is consumed by LLMSUT (which is a LLMServer wrapper).
    """
    print(dataset_cls)
    assert dataset_cls and issubclass(dataset_cls, LLMDataset), "dataset_cls my be a subclass of LLMDataset."

    class LLMQSL(dataset_cls):
        """ Mlperf-Inference QSL. LLMDataset subclass. """

        FILES = dataset_cls.FILES

        def __init__(self, sample_count: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sample_count = sample_count
            self.mlperf_qsl = lg.ConstructQSL(
                len(self),
                self.sample_count,

                # we load everything to host memory on init
                lambda _: None,
                lambda _: None,
            )

            self.logger.info("Initialized QSL.")
            self.logger.info(f'Total Sample Count: {len(self)}')
            self.logger.info(f'Performance Sample Count: {self.sample_count}')

        def __del__(self):
            lg.DestroyQSL(self.mlperf_qsl)
            self.logger.info("Destroyed QSL.")

    return LLMQSL

@add_prefix_logger()
class LLMSUTMultinode():
    """ Mlperf-Inference SUT. LLMServer subclass. """

    def __init__(self, dataset: LLMDataset, *args, **kwargs):
        TP = int(kwargs["trtllm_build_flags"]['tensor_parallelism'])
        PP = int(kwargs["trtllm_build_flags"]['pipeline_parallelism'])
        self.group_size = TP * PP
        self.dataset = dataset
        self.leader_ranks = [r for r in range(size) if r % self.group_size == 0 and r != size - 1]

        self.scheduled_tokens = {r: 0 for r in self.leader_ranks}
        self.token_cost = lambda q: len(q[1]) + kwargs["trtllm_runtime_flags"]['max_num_tokens']

        assert rank == size - 1
        self.sample_count = 0
        self.completed_samples = 0
        self.num_toks = 0
        self.pb = ProgressBar(len(self.leader_ranks), self.sample_count)

        self.mlperf_sut = lg.ConstructSUT(
            self.issue_queries, self.flush_queries
        )
        self.poll_for_completions()
        self.logger.info("Initialized SUT.")

    def __del__(self):
        self._stop_completion_thread()
        lg.DestroySUT(self.mlperf_sut)
        self.logger.info("Destroyed SUT.")

    def issue_queries_to_cores(self, query_samples: List[Tuple[int, List[int], List[int]]]):
        """Distribute query_samples across all TP-leader ranks."""

        samples_per_core = defaultdict(list)

        # token aware scheduling
        for q in query_samples:
            target_rank = min(self.scheduled_tokens, key=self.scheduled_tokens.get)
            samples_per_core[target_rank].append(q)
            self.scheduled_tokens[target_rank] += self.token_cost(q)

        dispatched = 0
        for rank, samples in samples_per_core.items():
            leader_rank = rank
            comm.isend(samples, dest=leader_rank, tag=TAG_BATCH).Wait()
            dispatched += len(samples)

        self.sample_count += dispatched

        self.pb.grow_total(self.sample_count)

    def run_test(self, test_settings, log_settings):
        print(f"Running test with settings: {test_settings}")
        print(f"Log settings: {log_settings}")
        lg.StartTestWithLogSettings(
            self.mlperf_sut,
            self.dataset.mlperf_qsl,
            test_settings,
            log_settings
        )
        print("Test completed. Waiting for completion...")

    ##### LLMServer overrides #####
    def flush_queries(self):

        """MLPerf harness calls this; must block until ALL cores drained."""

        # 1) Tell every leader rank to flush
        for r in self.leader_ranks:
            comm.isend(None, dest=r, tag=TAG_FLUSH).Wait()

        for r in self.leader_ranks:
            comm.recv(source=r, tag=TAG_FLUSH_ACK)

        print("All ranks flushed.")


    def issue_queries(self, query_samples: List[lg.QuerySample]):
        qsl_ids = [sample.id for sample in query_samples]
        qsl_indices = [sample.index for sample in query_samples]
        input_tokens = self.dataset.get_input_tokens(qsl_indices)
        stop_tokens = self.dataset.get_stop_tokens(qsl_indices)
        queries = list(zip(qsl_ids, input_tokens, stop_tokens))
        self.issue_queries_to_cores(queries)

    def poll_for_completions(self):
        """Start a background thread on rank-0 that forwards finished
        samples from remote ranks to MLPerf LoadGen."""

        def _rx():
            while not getattr(self, "_shutdown", False):
                # non-blocking probe keeps CPU usage low
                if comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_COMPLETE):
                    request_id, output_tokens, is_first_token = comm.recv(source=MPI.ANY_SOURCE,
                                                    tag=TAG_COMPLETE)
                    additional_stats, core_num = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_STATS)

                    if not is_first_token:
                        self.num_toks += len(output_tokens)
                        self.pb.step(1, len(output_tokens), additional_stats, core_num)

                    self.complete_request(request_id, output_tokens, is_first_token)

                else:
                    time.sleep(0.001)
            
            self.pb.close()

        t = threading.Thread(target=_rx, name="completion_rx", daemon=True)
        t.start()
        self._completion_thread = t

    def _stop_completion_thread(self):
        if hasattr(self, "_completion_thread"):
            self._shutdown = True
            self._completion_thread.join()
            print("Completion thread stopped.")

    @staticmethod
    def complete_request(request_id: int, output_tokens: List[int], is_first_token: bool):

        buf = np.ascontiguousarray(output_tokens, dtype=np.uint32)
        _live_buffers[request_id] = buf

        resp = lg.QuerySampleResponse(request_id,
                                    buf.ctypes.data,
                                    buf.nbytes,
                                    len(buf))
        if is_first_token:
            lg.FirstTokenComplete([resp])
        else:
            lg.QuerySamplesComplete([resp])

class LLMHarnessMultinode(BaseBenchmarkHarness):
    """Mlperf-Inference TRTLLM LLMHarness"""

    DATASET_CLS: type = LLMDataset
    CUSTOM_ARGS: List[str] = None

    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)

        harness_args = [
            "devices",
            "use_token_latencies",
            "enable_sort",
            "llm_gen_config_path",
            "trtllm_checkpoint_flags",
            "trtllm_build_flags",
            "trtllm_runtime_flags",
        ]

        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + harness_args
        if self.CUSTOM_ARGS is not None:
            self.flag_builder_custom_args += self.CUSTOM_ARGS

    def _get_harness_executable(self):
        """Return python command to LLMHarness runner python file"""
        return 'code/harness/harness_llm_py/runner.py'

    def _construct_terminal_command(self, argstr):
        cmd = f'python3 -m {self.executable.replace("/", ".").replace(".py", "")} {argstr}'
        return cmd

    def _get_engine_fpath(self, device_type, _, batch_size):
        # Override this function to pick up the right engine file
        if not self.default_engine_dir:
            return f"{self.engine_dir}/rank0.engine"

        tp_size = self.args['trtllm_build_flags']['tensor_parallelism']
        pp_size = self.args['trtllm_build_flags']['pipeline_parallelism']
        return f"{self.engine_dir}/{self.name}-{self.scenario.valstr()}-{device_type}-b{batch_size}-{self.precision}-tp{tp_size}pp{pp_size}-{self.config_ver}/rank0.engine"

    def _build_custom_flags(self, flag_dict):
        dataset_cls_fpath = inspect.getfile(self.DATASET_CLS)
        dataset_cls_path = dataset_cls_fpath.replace("/work/", "").replace(".py", "").replace("/", ".")
        dataset_cls = dataset_cls_path + f'.{self.DATASET_CLS.__name__}'

        def to_cli(value):
            match value:
                case bool() as b: return str(b).lower()
                case _: return str(value)

        flag_dict |= {
            key: ','.join(f"{k}:{to_cli(v)}" for k, v in value.items())
            for key, value in flag_dict.items()
            if key in ['trtllm_checkpoint_flags', 'trtllm_build_flags', 'trtllm_runtime_flags']
        }

        s = ' '.join([args_to_string(flag_dict),
                      f"--scenario {self.scenario.valstr()}",
                      f"--model {self.name}",
                      f"--dataset_cls {dataset_cls}"])
        return s

class ProgressBar:
    def __init__(self, num_leader_ranks: int, expected_total: int = 0):
        self.total        = expected_total
        self.done_samples = 0
        self.done_tokens  = 0
        self.t0           = time.time()

        # “–” for unknown until we see the first stats packet
        self.kv_util: list[str | float] = ["–"] * num_leader_ranks

        self.bar = tqdm(
            total         = self.total,
            unit          = "sample",
            dynamic_ncols = True,
            smoothing     = 0,
            leave         = False,
            bar_format    = (
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}] {postfix}"
            ),
        )
        self.bar.set_postfix_str(
            f"tok/s: 0.0  KV%: {' '.join(map(str, self.kv_util))}",
            refresh=False,
        )

    # -----------------------------------------------------------------
    def grow_total(self, new_total: int) -> None:
        if new_total > self.total:
            self.total       = new_total
            self.bar.total   = new_total
            self.bar.refresh()

    # -----------------------------------------------------------------
    def step(
        self,
        n_samples: int,
        n_tokens : int,
        additional_stats: dict[str, Any] | None = None,
        core_num: int | None = None,
    ) -> None:
        self.done_samples += n_samples
        self.done_tokens  += n_tokens
        self.bar.update(n_samples)

        # update per-core KV-util if we received it
        if additional_stats and core_num is not None:
            self.kv_util[core_num] = additional_stats.get("%kvcache_util", "–")

        tok_s  = f"{self.done_tokens / max(time.time() - self.t0, 1e-9):,.1f}"
        kv_str = " ".join(
            f"{v:>5.1f}%" if isinstance(v, (int, float)) else f"{v:>5}"
            for v in self.kv_util
        )

        self.bar.set_postfix_str(f"tok/s: {tok_s}  KV%: {kv_str}", refresh=False)

    # -----------------------------------------------------------------
    def close(self) -> None:
        self.bar.close()