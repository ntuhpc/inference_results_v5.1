import os
import argparse
from pathlib import Path
import glob
import shutil
import subprocess
import time

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)

loadgen_logfiles = [
    "mlperf_log_accuracy.json",
    "mlperf_log_detail.txt",
    "mlperf_log_summary.txt",
]

map_compliance_tests = {
    "mixtral-8x7b": ["TEST06"],
    "llama2-70b-99": ["TEST06"],
    "llama2-70b-99.9": ["TEST06"],
    "llama2-70b-interactive-99": ["TEST06"],
    "llama2-70b-interactive-99.9": ["TEST06"],
    "llama3.1-405b": ["TEST06"],
}


def code():
    pass


def make_directory(dirname):
    try:
        os.makedirs(dirname)
        logger.info(f"Created directory {dirname}")
    except FileExistsError as e:
        logger.info(e)


def copy_code(args, company_dir):
    input_dir = args.code_dir
    output_dir = f"{company_dir}/code/{args.benchmark}"

    make_directory(output_dir)

    shutil.copytree(
        input_dir,
        output_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            ".*", "__*", "inference_results_5.0", "submission"
        ),
    )


def copy_setup(args, company_dir):
    input_dir = args.setup_dir
    output_dir = f"{company_dir}/setup"

    make_directory(output_dir)

    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)


def copy_tools(args, company_dir):
    input_dir = args.tools_dir
    output_dir = f"{company_dir}/tools"

    make_directory(output_dir)

    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)


def copy_accuracy_logs(args, company_dir):
    for scenario in args.scenarios:
        input_dir = f"{args.input_dir}/{scenario}/accuracy"
        output_dir = f"{company_dir}/results/{args.system_name}/{args.benchmark}/{scenario}/accuracy"

        make_directory(output_dir)

        for logfile in loadgen_logfiles:
            shutil.copy(f"{input_dir}/{logfile}", output_dir)
        shutil.copy(f"{input_dir}/accuracy.txt", output_dir)


def copy_performance_logs(args, company_dir, iteration):
    for scenario in args.scenarios:
        input_dir = f"{args.input_dir}/{scenario}/performance"
        output_dir = f"{company_dir}/results/{args.system_name}/{args.benchmark}/{scenario}/performance"

        input_run_dir = f"{input_dir}/run_{str(iteration)}"
        output_run_dir = f"{output_dir}/run_{str(iteration)}"

        make_directory(output_run_dir)

        for logfile in loadgen_logfiles:
            shutil.copy(f"{input_run_dir}/{logfile}", output_run_dir)


def copy_compliance_logs(args, company_dir, tests):
    for scenario in args.scenarios:
        input_dir = f"{args.input_dir}/{scenario}/audit/compliance"
        output_dir = (
            f"{company_dir}/compliance/{args.system_name}/{args.benchmark}/{scenario}"
        )

        make_directory(output_dir)

        excluded_files = {
            "mlperf_log_detail.txt",
            "mlperf_log_summary.txt",
            "mlperf_log_trace.json",
        }

        for root, dirs, files in os.walk(input_dir):
            relative_path = os.path.relpath(root, input_dir)
            dest_path = os.path.join(output_dir, relative_path)

            os.makedirs(dest_path, exist_ok=True)

            for file in files:
                if file not in excluded_files:
                    shutil.copy2(
                        os.path.join(root, file), os.path.join(dest_path, file)
                    )


def copy_measurement_logs(args, company_dir, iteration):
    for scenario in args.scenarios:
        output_dir = (
            f"{company_dir}/measurements/{args.system_name}/{args.benchmark}/{scenario}"
        )

        make_directory(output_dir)

        shutil.copy(args.user_conf, f"{output_dir}/user.conf")
        shutil.copy(f"{args.mlperf_inference_dir}/mlperf.conf", output_dir)

        # subprocess.run(['touch', f"{output_dir}/README.md"])
        shutil.copy(
            f"{os.path.dirname(__file__)}/../README.md", f"{output_dir}/README.md"
        )
        shutil.copy(
            f"{os.path.dirname(__file__)}/measurements_system.json",
            f"{output_dir}/{args.system_name}.json",
        )


def setup_systems(args, company_dir, system_description):
    output_dir = f"{company_dir}/systems"
    make_directory(output_dir)
    shutil.copy(
        f"{os.path.dirname(__file__)}/{system_description}",
        f"{output_dir}/{args.system_name}.json",
    )


def exec_truncate_accuracy_logs(args):
    cmd = [
        "python",
        f"{args.mlperf_inference_dir}/tools/submission/truncate_accuracy_log.py",
        "--input",
        args.base_package_dir,
        "--submitter",
        args.company,
        "--backup",
        f"{args.base_package_dir}_bkp",
    ]
    subprocess.run(cmd)
    bkp_dir = f"{os.path.dirname(__file__)}/{args.base_package_dir}_bkp"
    if os.path.exists(bkp_dir):
        shutil.rmtree(bkp_dir)
    shutil.move(
        f"{args.base_package_dir}/{args.base_package_dir}_bkp",
        f"{os.path.dirname(__file__)}",
    )


def exec_submission_checker(args):
    cmd = [
        "python",
        f"{args.mlperf_inference_dir}/tools/submission/submission_checker.py",
        "--input",
        args.base_package_dir,
        "--version",
        args.mlperf_inference_version,
        "--submitter",
        args.company,
    ]
    subprocess.run(cmd)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system-description", type=str, default="dummy_system.json", help=""
    )
    parser.add_argument(
        "--mlperf-inference-dir", type=str, default="/app/mlperf_inference", help=""
    )
    parser.add_argument("--mlperf-inference-version", type=str, default="v5.0", help="")
    parser.add_argument(
        "--code-dir", type=str, default="/workspace/apps/mlperf", help=""
    )
    parser.add_argument(
        "--setup-dir", type=str, default="/lab-mlperf-inference/setup", help=""
    )  # TODO (peiran): we don't have setup now, but fix this in the future
    parser.add_argument(
        "--tools-dir", type=str, default="/workspace/apps/mlperf/tools", help=""
    )
    parser.add_argument("--input-dir", type=str, default=None, help="")
    parser.add_argument("--base-package-dir", type=str, default=None, help="")
    parser.add_argument("--division", type=str, default="closed", help="")
    parser.add_argument("--company", type=str, default="AMD", help="")
    parser.add_argument(
        "--scenarios", nargs="+", default=["Offline", "Server"], help=""
    )
    parser.add_argument("--system-name", type=str, default=None, help="")
    parser.add_argument("--benchmark", type=str, default="llama2-70b-99.9", help="")
    parser.add_argument(
        "--user-conf",
        type=str,
        default=None,
        help="Path to the user.conf file used for the submission",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    print(f"scenarios={args.scenarios}")

    company_dir = f"{args.base_package_dir}/{args.division}/{args.company}"
    compliance_tests = map_compliance_tests[args.benchmark]

    system_description = args.system_description

    setup_systems(args, company_dir, system_description)
    copy_code(args, company_dir)
    # copy_setup(args, company_dir)  # FIXME: (peiran) currently, we don't have setup folder
    copy_tools(args, company_dir)
    copy_accuracy_logs(args, company_dir)
    copy_performance_logs(args, company_dir, 1)
    copy_compliance_logs(args, company_dir, compliance_tests)
    copy_measurement_logs(args, company_dir, 1)

    exec_truncate_accuracy_logs(args)
    time.sleep(10)
    exec_submission_checker(args)


if __name__ == "__main__":
    main()
