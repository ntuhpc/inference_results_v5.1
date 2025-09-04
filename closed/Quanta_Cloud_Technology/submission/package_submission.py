import os
import argparse
from pathlib import Path
import glob
import shutil
import subprocess
import time
import json

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__file__)

loadgen_logfiles = [
    "mlperf_log_accuracy.json",
    "mlperf_log_detail.txt",
    "mlperf_log_summary.txt"
]

map_compliance_tests = {
    "mixtral-8x7b": ["TEST06"],
    "llama2-70b-99": ["TEST06"],
    "llama2-70b-99.9": ["TEST06"],
    "llama3.1-405b": ["TEST06"]
}

map_accuracy_scripts = {
    "mixtral-8x7b": "check_mixtral_accuracy_scores",
    "llama2-70b-99": "check_llama2_accuracy_scores",
    "llama2-70b-99.9": "check_llama2_accuracy_scores",
    "llama3.1-405b": "check_llama3_accuracy_scores"
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

    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.*', '__*'))

def copy_setup(args, company_dir):
    input_dir = args.setup_dir
    output_dir = f"{company_dir}/setup"

    make_directory(output_dir)

    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns('vllm', 'aiter'))

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
        input_dir = f"{args.input_dir}/{scenario}/audit/compliance/TEST06/accuracy"
        output_dir_base = f"{company_dir}/compliance/{args.system_name}/{args.benchmark}/{scenario}/TEST06"
        output_dir_accuracy = f"{output_dir_base}/accuracy"

        make_directory(output_dir_accuracy)
        shutil.copytree(input_dir, output_dir_accuracy, dirs_exist_ok=True)

        accuracy_file = f"{args.input_dir}/{scenario}/audit/compliance/TEST06/verify_accuracy.txt"
        shutil.copy(accuracy_file, output_dir_base)


def get_model_name(arg_benchmark):
    # Strip off the accuracy part from benchmark in case of Llama2-70b-99.9
    if arg_benchmark.startswith('llama2-70b'):
        model_name = arg_benchmark.rsplit('-', 1)[0]
    else:
        model_name = arg_benchmark

    return model_name.replace('-', '_')

def get_gpu_count_from_system_name(arg_system_name):
    return arg_system_name.split('x')[0]


def get_gpu_name_from_system_name(arg_system_name):
    return arg_system_name.split('x')[1].split('_')[0].lower()


def update_measurement_readme(args, scenario, file):
    # Remove the second part of the file
    with open(file, 'r') as f:
        lines = f.readlines()

    target_line = "### Running the benchmark and submission packaging\n"

    try:
        target_index = lines.index(target_line)
    except ValueError:
        print(f"Line '{target_line}' not found in the file.")
        return

    new_lines = lines[:target_index]

    with open(file, 'w') as f:
        f.writelines(new_lines)

    # Add second part with commands to run the benchmark
    with open(f"{os.path.dirname(__file__)}/README_cmds.md", 'r') as f:
        second_part = f.readlines()

    replacements = {
        'MODEL_NAME': get_model_name(args.benchmark).replace('_', '-'),
        'BENCHMARK_NAME': args.benchmark,
        'GPU_NAME': get_gpu_name_from_system_name(args.system_name),
        'GPU_COUNT': get_gpu_count_from_system_name(args.system_name),
        'USER_CONF': os.path.basename(args.user_conf),
        'scenario': scenario.lower(),
        'SCENARIO': scenario,
        'ACCURACY_SCRIPT_NAME': map_accuracy_scripts[args.benchmark]
    }

    for old_string, new_string in replacements.items():
        second_part = [line.replace(old_string, new_string) for line in second_part]

    with open(file, 'a') as f:
        f.writelines(second_part)


def copy_measurement_logs(args, company_dir, iteration):
    # Get the model name to copy the proper README file
    model_name = get_model_name(args.benchmark)

    for scenario in args.scenarios:
        output_dir = f"{company_dir}/measurements/{args.system_name}/{args.benchmark}/{scenario}"

        make_directory(output_dir)

        output_user_conf = f"{output_dir}/user.conf"
        shutil.copy(args.user_conf, output_user_conf)
        shutil.copy(f"{args.mlperf_inference_dir}/mlperf.conf", output_dir)

        output_readme = f"{output_dir}/README.md"
        shutil.copy(f"{os.path.dirname(__file__)}/README_{model_name}.md", output_readme)
        update_measurement_readme(args, scenario, output_readme)
        shutil.copy(f"{os.path.dirname(__file__)}/measurements_system.json", f"{output_dir}/{args.system_name}.json")


def copy_documentation(args, company_dir):
    file_names = ["bandwidth.md", "calibration.md"]
    input_dir = os.path.dirname(__file__)
    output_dir = f"{company_dir}/documentation"

    make_directory(output_dir)
    for file_name in file_names:
        shutil.copy(f"{input_dir}/{file_name}", f"{output_dir}/{file_name}")


def setup_systems(args, company_dir):
    output_dir = f"{company_dir}/systems"
    make_directory(output_dir)
    shutil.copy(f"{os.path.dirname(__file__)}/{args.system_json}", f"{output_dir}/{args.system_name}.json")
    with open(f"{output_dir}/{args.system_name}.json", 'r+') as file:
        data = json.load(file)
        data["submitter"] = args.company
        file.seek(0)
        json.dump(data, file, indent=4)


def exec_truncate_accuracy_logs(args):
    cmd = [
        'python', f"{args.mlperf_inference_dir}/tools/submission/truncate_accuracy_log.py",
        '--input', args.base_package_dir,
        '--submitter', args.company,
        '--backup', f"{args.base_package_dir}_bkp"
    ]
    subprocess.run(cmd)
    bkp_dir = f"{os.path.dirname(__file__)}/{args.base_package_dir}_bkp"
    if os.path.exists(bkp_dir):
        shutil.rmtree(bkp_dir)
    shutil.move(f"{args.base_package_dir}/{args.base_package_dir}_bkp", f"{os.path.dirname(__file__)}")


def exec_submission_checker(args):
    cmd = [
        'python', f"{args.mlperf_inference_dir}/tools/submission/submission_checker.py",
        '--input', args.base_package_dir,
        '--version', args.mlperf_inference_version,
        '--submitter', args.company
    ]
    subprocess.run(cmd)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-inference-dir", type=str, default="/app/mlperf_inference", help="")
    parser.add_argument("--mlperf-inference-version", type=str, default="v5.1", help="")
    parser.add_argument("--code-dir", type=str, default="/lab-mlperf-inference/code", help="")
    parser.add_argument("--setup-dir", type=str, default="/lab-mlperf-inference/setup", help="")
    parser.add_argument("--tools-dir", type=str, default="/lab-mlperf-inference/tools", help="")
    parser.add_argument("--input-dir", type=str, default=None, help="")
    parser.add_argument("--base-package-dir", type=str, default=None, help="")
    parser.add_argument("--division", type=str, default="closed", help="")
    parser.add_argument("--company", type=str, default="AMD", help="")
    parser.add_argument("--scenarios", nargs="+", default=["Offline", "Server", "Interactive"], help="")
    parser.add_argument("--system-name", type=str, default=None, help="")
    parser.add_argument("--benchmark", type=str, default="llama2-70b-99.9", help="")
    parser.add_argument("--user-conf", type=str, default=None, help="Path to the user.conf file used for the submission")
    parser.add_argument("--system-json", type=str, default="dummy_system.json", help="Contains machine specifications")
    args = parser.parse_args()
    return args


def main(args):
    print(f"scenarios={args.scenarios}")

    company_dir = f"{args.base_package_dir}/{args.division}/{args.company}"
    compliance_tests = map_compliance_tests[args.benchmark]

    setup_systems(args, company_dir)
    copy_code(args, company_dir)
    copy_setup(args, company_dir)
    copy_documentation(args, company_dir)
    copy_accuracy_logs(args, company_dir)
    copy_performance_logs(args, company_dir, 1)
    copy_compliance_logs(args, company_dir, compliance_tests)
    copy_measurement_logs(args, company_dir, 1)

    exec_truncate_accuracy_logs(args)
    time.sleep(10)
    exec_submission_checker(args)


if __name__ == "__main__":
    main(get_args())
