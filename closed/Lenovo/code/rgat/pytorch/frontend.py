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


from code.common import logging, dict_get, run_command, args_to_string
from code.common.constants import Benchmark, Scenario
from code.common.harness import *
from code.common.submission import generate_measurements_entry
from code.common.systems.system_list import DETECTED_SYSTEM

import code.common.arguments as common_args
import io
import os
import subprocess
import sys


class RGATHarness:

    def __init__(self, args, benchmark):
        self.args = args
        self.benchmark = benchmark
        self.scenario = args["scenario"]

        assert self.benchmark == Benchmark.RGAT
        assert self.scenario == Scenario.Offline

        self.name = self.benchmark.valstr()
        self.workload_settings = dict_get(args, "workload_setting", default=None)
        self.system_id = args["system_id"]
        self.config_ver = args["config_ver"]

        self.executable = "python3 -m code.rgat.pytorch.harness"
        self.env_vars = os.environ.copy()

        # Do we really need this??
        self.vboost_slider = 0
        if "is_hopper" in DETECTED_SYSTEM.extras["tags"]:
            self.vboost_slider = dict_get(self.args, 'vboost_slider', 0)  # always reset to default when not provided by the user

    def get_system_name(self, add_trt=True):
        override_system_name = dict_get(self.args, "system_name", default=None)
        if override_system_name not in {None, ""}:
            return override_system_name

        system_name = self.system_id
        if add_trt:
            system_name = f"{system_name}_TRT"
        return system_name

    def _get_submission_benchmark_name(self):
        return "rgat"

    def get_full_log_dir(self):
        return os.path.join(self.args["log_dir"],
                            self.get_system_name(),
                            self._get_submission_benchmark_name(),
                            self.scenario.valstr())

    def build_non_custom_flags(self):
        return self.build_flags()

    def build_flags(self):
        flag_dict = dict()

        # Generate flags for logfile names.
        log_dir = self.get_full_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        flag_dict["logfile_outdir"] = log_dir
        flag_dict["logfile_prefix"] = "mlperf_log_"

        # Handle performance sample count
        perf_sample_count = dict_get(self.args, "performance_sample_count", None)
        perf_sample_count_override = dict_get(self.args, "performance_sample_count_override", None)
        if perf_sample_count_override is not None:
            flag_dict["performance_sample_count"] = perf_sample_count_override
        elif perf_sample_count is not None:
            flag_dict["performance_sample_count"] = perf_sample_count
        elif benchmark_qsl_size_map[self._get_submission_benchmark_name()] > 0:
            flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self._get_submission_benchmark_name()]
        else:
            flag_dict["performance_sample_count"] = self.args["gpu_batch_size"]

        # Only offline is supported
        scenario_keys = common_args.getScenarioMetricArgs(Scenario.Offline)
        for arg in scenario_keys:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        # MLPINF-853: Special handing of --test_run. Use min_duration=60000 and min_query_count=1.
        if flag_dict.get("test_run", False):
            if "min_duration" not in flag_dict:
                flag_dict["min_duration"] = 60000
            if "min_query_count" not in flag_dict:
                flag_dict["min_query_count"] = 1
            flag_dict["test_run"] = None
        if "min_duration" in flag_dict:
            logging.info(f"min_duration is overwritten to {flag_dict['min_duration']}.")
        if "min_query_count" in flag_dict:
            logging.info(f"min_query_cnt is overwritten to {flag_dict['min_query_count']}.")

        flag_dict["gpu_batch_size"] = self.args["gpu_batch_size"]["rgat"],
        flag_dict["test_mode"] = self.args.get("test_mode", "PerformanceOnly")
        return flag_dict

    def run_harness(self,
                    flag_dict=None,
                    skip_generate_measurements=False,
                    use_py_harness=False):
        if flag_dict is None:
            flag_dict = self.build_flags()

        if not skip_generate_measurements:
            generate_measurements_entry(
                self.get_system_name(),
                self.name,
                self._get_submission_benchmark_name(),
                self.scenario,
                "int64",
                "fp16",
                flag_dict)

        subset = {k: v
                  for k, v in flag_dict.items()
                  if k in ["gpu_batch_size",
                           "test_mode",
                           "mlperf_conf_path",
                           "user_conf_path",
                           "performance_sample_count",
                           "logfile_outdir",
                           "shm_buffer_size"]}
        # Just in case batch_size is turned into a tuple by GBS processing
        if isinstance(subset["gpu_batch_size"], tuple):
            subset["gpu_batch_size"] = subset["gpu_batch_size"][0]
        argstr = args_to_string(subset)

        def set_vboost(value: int = 0):
            logging.info(f"setting vboost to {value if value != 0 else 'gpu default'}")
            run_command(f"sudo nvidia-smi boost-slider --vboost {value}")

        set_vboost(self.vboost_slider)

        cmd = f"{self.executable} {argstr}"
        logging.info(f"Running command: {cmd}")
        p = subprocess.Popen(cmd,
                             shell=True,
                             env=self.env_vars)
        if (retcode := p.wait()) != 0:
            raise subprocess.CalledProcessError(retcode, cmd)

        set_vboost(0)

        # Return harness result.
        scenario_key = get_loadgen_log_keys(self.scenario, self.benchmark)
        query_metric_key = QUERY_METRIC_CONSTRAINTS[self.scenario.valstr()][0]
        loadgen_query_keys = ["result_validity", scenario_key, "early_stopping_met", query_metric_key, "effective_min_duration_ms"]

        results = from_loadgen_by_keys(os.path.join(self.args["log_dir"],
                                                    self.get_system_name(),
                                                    self._get_submission_benchmark_name(),
                                                    self.scenario.valstr()),
                                       loadgen_query_keys)

        _got = float(results.get(query_metric_key, "0.0"))
        _thresh = QUERY_METRIC_CONSTRAINTS[self.scenario.valstr()][1]
        satisfies_query_constraint = (_got >= _thresh)

        test_mode = flag_dict.get("test_mode", "PerformanceOnly")
        results.update({"system_name": self.get_system_name(),
                        "benchmark_short": self.benchmark.valstr(),
                        "benchmark_full": self._get_submission_benchmark_name(),
                        "scenario": self.scenario.valstr(),
                        "test_mode": test_mode,
                        "tensorrt_version": trt.__version__,
                        "detected_system": DETECTED_SYSTEM.summary_description(),
                        "scenario_key": scenario_key,
                        "satisfies_query_constraint": satisfies_query_constraint
                        })
        return results
