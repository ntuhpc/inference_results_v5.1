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

import mlperf_loadgen as lg
import argparse
import os
import logging
import sys
import requests
import time

sys.path.insert(0, os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d UTC - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.Formatter.converter = time.gmtime
log = logging.getLogger("Llama-8B-MAIN")

# function to check the model name in server matches the user specified one


def verify_model_name(user_specified_name, url):
    response = requests.get(url)
    if response.status_code == 200:
        response_dict = response.json()
        server_model_name = response_dict["data"][0]["id"]
        if user_specified_name == server_model_name:
            return {"matched": True, "error": False}
        else:
            return {
                "matched": False,
                "error": f"User specified {user_specified_name} and server model name {server_model_name} mismatch!",
            }
    else:
        return {
            "matched": False,
            "error": f"Failed to get a valid response. Status code: {response.status_code}",
        }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["Offline", "SingleStream"],
        default="Offline",
        help="Scenario",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="")
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Run accuracy mode")
    parser.add_argument(
        "--audit-conf",
        type=str,
        default="audit.conf",
        help="audit config for LoadGen settings during compliance runs",
    )
    parser.add_argument(
        "--user-conf",
        type=str,
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    # TODO: This interpretation of 'total-sample-count' is a little
    # misleading. Fix it
    parser.add_argument(
        "--total-sample-count",
        type=int,
        default=13368,
        help="Number of samples to use in benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Model batch-size to use in benchmark.",
    )
    parser.add_argument(
        "--output-log-dir", type=str, default="output-logs", help="Where logs are saved"
    )
    parser.add_argument(
        "--enable-log-trace",
        action="store_true",
        help="Enable log tracing. This file can become quite large",
    )
    parser.add_argument(
        "--lg-model-name",
        type=str,
        default="llama3_1-8b-edge",
        choices=["llama3_1-8b-edge"],
        help="Model name(specified in llm server)",
    )
    parser.add_argument(
        "--token-output-file",
        type=str,
        default="tokens_output.json",
        help="Output file for token logging (JSON format)",
    )
    parser.add_argument(
        "--engine-dir",
        type=str,
        default="",
        help="Engine directory",
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Use asynchronous mode",
    )

    args = parser.parse_args()
    return args


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    "singlestream": lg.TestScenario.SingleStream,
}


def main():
    args = get_args()
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario.lower()]
    # mlperf.conf is automatically loaded by the loadgen
    # settings.FromConfig(args.mlperf_conf, "llama3_1-8b", args.scenario)
    settings.FromConfig(args.user_conf, args.lg_model_name, args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    from SUT_DRIVE import SUT

    sut_map = {"offline": SUT, "singlestream": SUT}

    sut_cls = sut_map[args.scenario.lower()]

    sut = sut_cls(
        model_path=args.model_path,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path,
        total_sample_count=args.total_sample_count,
        scenario=args.scenario.lower(),
        token_output_file=args.token_output_file,
        engine_dir=args.engine_dir,
        async_mode=args.async_mode
    )

    # Start sut before loadgen starts
    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    

    log.info("Starting Benchmark run")
    lg.StartTestWithLogSettings(
        lgSUT,
        sut.qsl,
        settings,
        log_settings,
        args.audit_conf)
    log.info("Run Completed!")

    log.info("Cleaning up...")
    sut.stop()
    
    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
