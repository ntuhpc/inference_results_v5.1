# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import mlperf_loadgen as lg
import os
import logging

os.environ["DGLBACKEND"] = "pytorch"

from code.common.harness import benchmark_qsl_size_map, BaseBenchmarkHarness
from nvmitten.nvidia.cupy import CUDARTWrapper as cudart
from pathlib import Path
from typing import List, Optional, Dict

from .rgat import RGATOfflineServer
from .config import RGATConfig


test_mode_map = {
    "PerformanceOnly": lg.TestMode.PerformanceOnly,
    "AccuracyOnly": lg.TestMode.AccuracyOnly,
    "SubmissionRun": lg.TestMode.SubmissionRun,
}

log_mode_map = {
    "AsyncPoll": lg.LoggingMode.AsyncPoll,
    "EndOfTestOnly": lg.LoggingMode.EndOfTestOnly,
    "Synchronous": lg.LoggingMode.Synchronous,
}


class RGATLoadgenRunner:
    def __init__(self,
                 devices: List[int],
                 batch_size: int,
                 test_mode: str,
                 logfile_outdir: os.PathLike,
                 mlperf_conf_path: os.PathLike,
                 user_conf_path: os.PathLike,
                 performance_sample_count: int = 788379,
                 buffer_size: int = 788379 * 2,
                 num_complete_threads: int = 128,
                 log_mode: str = "AsyncPoll",
                 log_poll_interval_ms: int = 1000,
                 enable_log_trace: bool = False,
                 logfile_prefix: str = "mlperf_log_",
                 logfile_suffix: str = ""):
        test_settings = lg.TestSettings()
        test_settings.scenario = lg.TestScenario.Offline  # GNN only supports Offline
        test_settings.mode = test_mode_map[test_mode]

        logging.info(f"Loading mlperf.conf from {mlperf_conf_path}")
        test_settings.FromConfig(mlperf_conf_path, "rgat", "Offline", 2)
        logging.info(f"Loading user.conf from {user_conf_path}")
        test_settings.FromConfig(user_conf_path, "rgat", "Offline", 1)
        test_settings.server_coalesce_queries = True

        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = logfile_outdir
        log_output_settings.prefix = logfile_prefix
        log_output_settings.suffix = logfile_suffix
        log_output_settings.copy_summary_to_stdout = True
        Path(log_output_settings.outdir).mkdir(parents=True, exist_ok=True)

        log_settings = lg.LogSettings()
        log_settings.log_output = log_output_settings
        log_settings.log_mode = log_mode_map[log_mode]
        log_settings.log_mode_async_poll_interval_ms = log_poll_interval_ms
        log_settings.enable_trace = enable_log_trace

        # QSL - Dataset and loading is handled by wholegraph, which is part of the RGAT server. Pass a no-op to LG
        # We don't want to load extraneous memory. This number is obtained from:
        # torch.load("/home/mlperf_inf_rgat/optimized/converted/graph/full/val_idx.pt").size()
        total_count = 788379
        qsl = lg.ConstructQSL(total_count,
                              performance_sample_count,
                              lambda _: None,
                              lambda _: None)

        # Create server
        serv = RGATOfflineServer(devices,
                                 RGATConfig(batch_size=batch_size),
                                 ds_size=total_count,
                                 delegator_max_size=buffer_size,
                                 n_complete_threads=num_complete_threads)
        sut = lg.ConstructSUT(serv.issue_queries, serv.flush_queries)

        lg.StartTestWithLogSettings(sut, qsl, test_settings, log_settings)


def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', '1'):
            return True
        elif v.lower() in ('false', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_batch_size", type=int, default=6144, help="Max Batch size to use for gpu end-to-end inference")
    parser.add_argument("--test_mode", choices=["PerformanceOnly", "AccuracyOnly", "SubmissionRun"], default="PerformanceOnly")
    parser.add_argument("--logfile_outdir", help="Path to store mlperf logs")
    parser.add_argument("--mlperf_conf_path", help="Path to mlperf.conf")
    parser.add_argument("--user_conf_path", help="Path to user.conf")
    parser.add_argument("--performance_sample_count", type=int, default=788379, help="Performance sample count for test")
    parser.add_argument("--log_mode", type=str, default="AsyncPoll", help="Logging mode for Loadgen")
    parser.add_argument("--enable_log_trace", action="store_true", help="Generate mlperf_log_trace.json")
    parser.add_argument("--log_mode_async_poll_interval_ms", type=int, default=1000, help="Specify the poll interval for asynchrounous logging")

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    args = get_args()
    device_count = cudart.cudaGetDeviceCount()

    harness = RGATLoadgenRunner(
        list(range(device_count)),
        args.gpu_batch_size,
        args.test_mode,
        args.logfile_outdir,
        args.mlperf_conf_path,
        args.user_conf_path,
        performance_sample_count=args.performance_sample_count,
        log_mode=args.log_mode,
        log_poll_interval_ms=args.log_mode_async_poll_interval_ms,
        enable_log_trace=args.enable_log_trace)
