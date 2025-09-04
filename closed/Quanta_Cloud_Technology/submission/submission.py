#!/usr/bin/python3

import os
import shutil
import datetime
from dataclasses import dataclass
import json
from omegaconf import OmegaConf
from functools import wraps
import multiprocessing as mp
import textwrap

MIXTRAL = "mixtral-8x7b"
LLAMA2 = "llama2-70b"
SUPPORTED_MODELS = [MIXTRAL, LLAMA2]
SUPPORTED_SCENARIOS = ['Offline', 'Server', 'Interactive']
CACHE = 'cache/'
EXPERIMENTS = CACHE + 'experiments'
BEST_RESULTS = CACHE + 'current-best'
BEST_RESULT_PERFORMANCE_SUB_FOLDER = 'performance/run_1'
BEST_RESULT_ACCURACY_SUB_FOLDER = 'accuracy/'
BEST_RESULT_COMPLIANCE_SUB_FOLDER = 'audit/compliance/'
MODEL_CONF_NAME_WO_EXT = 'model'
MODEL_CONF_NAME = f'{MODEL_CONF_NAME_WO_EXT}.yaml'
USER_CONF_NAME = 'user.conf'
TEST06_DIR = "/app/mlperf_inference/compliance/nvidia/TEST06"
AUDIT_CONF = 'audit.config'
CODE_DIR = f'{os.path.dirname(__file__)}/../code/'
SUBMISSION_PACKAGE_NAME = 'inference_results_5.1'
TEST=bool(int(os.environ.get("TEST", 0)))
DEBUG=bool(int(os.environ.get("DEBUG", 0)))

def debug(func):
    if DEBUG:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            print(f"Function '{func.__name__} with {args=} {kwargs=}' returned: {result}")
            return result
        return wrapper
    return func

@dataclass
class ExperimentResult:
    is_valid : bool = False
    requested_scenario : str = ''
    # Offline
    tokens_per_sec : float = 0
    samples_per_second : float = 0
    # Server
    completed_samples_per_second : float = 0
    completed_tokens_per_second : float = 0
    ttft : int = 0
    tpot : int = 0

@dataclass
class ExperimentAccuracy:
    is_valid : bool = False
    requested_model : str = ''
    # Llama2
    rouge1 : float = 0
    rouge2 : float = 0
    rougeL : float = 0
    tokens_per_sample : float = 0
    # Mixtral
    gsm8k : float = 0
    mbxp : float = 0

@dataclass
class ExperimentCompliance:
    is_valid : bool = False

map_accuracy_limits = {
    LLAMA2: {'rouge1': 44.3867688, 'rouge2': 22.0131648, 'rougeL': 28.5875838, 'tokens_per_sample': 265.005},
    MIXTRAL: {'rouge1': 45.14291, 'rouge2': 23.11907, 'rougeL': 30.15619, 'gsm8k': 72.9234, 'mbxp': 59.5584, 'tokens_per_sample': 130.356},
}

map_accuracy_upper_limits = {
    LLAMA2: {'tokens_per_sample': 323.895},
    MIXTRAL: {'tokens_per_sample': 160.49},
}

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='MLPerf Inference Submission Tool',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', required=True, choices=SUPPORTED_MODELS)
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Status command
    status_parser = subparsers.add_parser('status', help='Check best model state')

    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run an experiment with the specified model and scenario',
                                       formatter_class=argparse.RawTextHelpFormatter)
    exp_parser.add_argument('--scenario', required=True, choices=SUPPORTED_SCENARIOS)
    exp_parser.add_argument('--model-conf', required=True, help=textwrap.dedent('''\
                            Path to the model configuration YAML file
                            (e.g. code/harness_llm/models/llama2-70b/offline_mi325x.yaml).
                            '''))
    exp_parser.add_argument('--user-conf', required=True, help=textwrap.dedent('''\
                            Path to the user configuration file
                            (e.g. code/user_mi325x.conf).
                            '''))

    # Update best command
    update_parser = subparsers.add_parser('update_best', help='Select the current best result for the specified model and scenario')
    update_parser.add_argument('--scenario', required=True, choices=SUPPORTED_SCENARIOS)

    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Run accuracy or compliance')
    prepare_parser.add_argument('mode', choices=['accuracy', 'compliance'])
    prepare_parser.add_argument('--scenario', required=True, choices=SUPPORTED_SCENARIOS)
    prepare_parser.add_argument('--force', action='store_true', help='Overwrite existing valid result')

    # Package command
    package_parser = subparsers.add_parser('package',
                                           help=textwrap.dedent('''\
                                            Package the current best results for submission.
                                            It expects everything ready, check status before calling.
                                            You have to set environment variables:
                                            GPU_COUNT, GPU_NAME, CPU_COUNT, CPU_NAME, COMPANY.
                                            The package will be created in the current directory.
                                            '''))

    return parser.parse_args()

@debug
def run_cmd(cmd):
    from subprocess import run
    return run(cmd, capture_output=True, text=True)

@debug
def power_setting(model, scenario):
    gpu = run_cmd(['bash', f'{CODE_DIR}/scripts/determine_accelerator.sh']).stdout.rstrip('\n')
    result = run_cmd(['bash', f'{CODE_DIR}/scripts/power_settings.sh', model, gpu, scenario.lower()])
    return result


def create_harness_config(config_path, user_conf_path, output_log_dir, config_name=MODEL_CONF_NAME_WO_EXT, user_conf_name=USER_CONF_NAME, test_mode='performance'):
    import sys; sys.path.insert(0, CODE_DIR)
    from harness_llm.common.config_parser import HarnessCfg
    args = [
        f"config_path={config_path}",
        f"config_name={config_name}",
        f"test_mode={test_mode}",
        f"harness_config.user_conf_path={user_conf_path}/{user_conf_name}",
        f"harness_config.output_log_dir={output_log_dir}",
    ]
    if TEST:
        args.extend([
            f"harness_config.duration_sec=30",
            f"harness_config.total_sample_count=16" if next(s for s in SUPPORTED_SCENARIOS if s in output_log_dir) == 'Offline' else '',
        ])

    return HarnessCfg().create(OmegaConf.from_dotlist(args))


@debug
def get_actual_model(model, scenario):
    if 'Interactive' == scenario:
        assert model == LLAMA2, f"Interactive only supported for {LLAMA2}"
        return f"{LLAMA2}-interactive"
    else:
        return model


@debug
def get_actual_scenario(scenario):
    return 'Server' if 'Interactive' == scenario else scenario


@debug
def is_exp_valid(exp):
    return (exp and exp.is_valid)


def check_user_config(user_conf):
    assert os.path.exists(user_conf), f"File not found: {user_conf}"

    with open(user_conf, 'r') as conf:
        content = conf.read()

    if not content:
        raise RuntimeError(f"User config is empty: {user_conf}")

    if "min_duration" not in content:
        raise RuntimeError("User config does not contain 'min_duration'")

    if "target_qps" not in content:
        raise RuntimeError("User config does not contain 'target_qps'")


@debug
def is_exp_better(exp, best, scenario):
    assert exp.requested_scenario == best.requested_scenario == scenario, f"{exp.requested_scenario} == {best.requested_scenario} == {scenario}"
    if scenario == 'Offline':
        return best.tokens_per_sec < exp.tokens_per_sec
    if scenario in ['Server', 'Interactive']:
        return best.completed_tokens_per_second < exp.completed_tokens_per_second
    raise ValueError(f"Unknown scenario: {scenario}")


def update_best_dir(exp_base_dir, best_base_dir):
    print("Found better result, updating current-best")
    results_dir = f"{exp_base_dir}/results"
    configs_dir = f"{exp_base_dir}/configs"
    best_res_dir = f"{best_base_dir}/{BEST_RESULT_PERFORMANCE_SUB_FOLDER}"
    shutil.rmtree(best_res_dir, ignore_errors=True)
    shutil.copytree(results_dir, best_res_dir, dirs_exist_ok=True)
    shutil.copytree(configs_dir, best_base_dir, dirs_exist_ok=True)


def experiment(model, scenario, model_config, user_config):
    print(f"Running experiment with {model=} {scenario=} {model_config=} {user_config=}")
    import sys; sys.path.insert(0, CODE_DIR)
    from main import run_mlperf_tests

    check_user_config(user_config)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{EXPERIMENTS}/{model}/{scenario}/{timestamp}"
    results_dir = f"{output_dir}/results"
    os.makedirs(results_dir)
    configs_dir = f"{output_dir}/configs"
    os.makedirs(configs_dir)

    model_conf_path = os.path.join(configs_dir, MODEL_CONF_NAME)
    user_conf_path = os.path.join(configs_dir, USER_CONF_NAME)

    shutil.copy(os.path.abspath(model_config), model_conf_path)
    shutil.copy(os.path.abspath(user_config), user_conf_path)

    actual_scenario = get_actual_scenario(scenario)
    actual_model = get_actual_model(model, scenario)

    with open(user_config, 'r') as user_conf:
        with open(user_conf_path, 'w') as user_conf_final:
            for line in user_conf.readlines():
                if line.startswith(f"{actual_model}.*.") or line.startswith(f"{actual_model}.{actual_scenario}."):
                    user_conf_final.write(line)

    harnessCfg = create_harness_config(config_path=configs_dir,
                                       config_name=MODEL_CONF_NAME_WO_EXT,
                                       user_conf_path=configs_dir,
                                       user_conf_name=USER_CONF_NAME,
                                       output_log_dir=results_dir,
                                       test_mode="performance")

    assert harnessCfg.benchmark_name == actual_model, f"Parameter model and config benchmark_name are different: {actual_model} vs {harnessCfg.benchmark_name}"
    assert harnessCfg.scenario == actual_scenario.lower(), f"Parameter scenario and config scenario are different: {actual_scenario.lower()} vs {harnessCfg.scenario}"
    assert harnessCfg.test_mode == "performance", f"test_mode should be 'performance', actual is '{harnessCfg.test_mode}'"
    assert harnessCfg.backend == "vllm", f"backend should be 'vllm', actual is '{harnessCfg.backend}'"

    print(f"Starting MLPerf run")
    power_setting(actual_model, scenario)
    print("Running MLPerf perf...")
    run_mlperf_tests(harnessCfg)

    best_res_base_dir = f"{BEST_RESULTS}/{model}/{scenario}"
    best_res_dir = f"{best_res_base_dir}/{BEST_RESULT_PERFORMANCE_SUB_FOLDER}"

    exp = get_result(results_dir)
    best = get_result(best_res_dir)

    if is_exp_valid(exp) and (not best or is_exp_better(exp, best, actual_scenario)):
        update_best_dir(output_dir, best_res_base_dir)
    else:
        print(f"Result was not used (it was {'valid' if is_exp_valid(exp) else 'invalid'}).")


def update_best(model, scenario):
    exp_dir = f"{EXPERIMENTS}/{model}/{scenario}"
    if not os.path.exists(exp_dir):
        raise RuntimeError(f"No experiments found for model {model} and scenario {scenario}.")
    timestamps = os.listdir(exp_dir)
    best_res_base_dir = f"{BEST_RESULTS}/{model}/{scenario}"
    best_res_dir = f"{best_res_base_dir}/{BEST_RESULT_PERFORMANCE_SUB_FOLDER}"
    actual_scenario = get_actual_scenario(scenario)

    best_res = None
    best_timestamp = None
    if os.path.exists(best_res_dir):
        best_res = get_result(best_res_dir)

    for timestamp in timestamps:
        exp_res = get_result(os.path.join(exp_dir, timestamp, 'results'))
        if not is_exp_valid(exp_res):
            continue

        if not best_res or is_exp_better(exp_res, best_res, actual_scenario):
            best_res = exp_res
            best_timestamp = timestamp

    if best_timestamp:
        update_best_dir(f"{exp_dir}/{best_timestamp}", best_res_base_dir)
    else:
        print(f"No better experiments found in {exp_dir}. Skipping update.")


def prepare(model, mode, scenario, force=False):
    print(f"Running prepare with {model=} {mode=} {force=}")
    import sys; sys.path.insert(0, CODE_DIR)
    from main import run_mlperf_tests

    result = get_result(folder=f"{BEST_RESULTS}/{model}/{scenario}/{BEST_RESULT_PERFORMANCE_SUB_FOLDER}")
    if result is None:
        print(f"{scenario} not exists, skipping it.")
    assert result.is_valid
    best_res_base_dir = f"{BEST_RESULTS}/{model}/{scenario}"
    actual_scenario = get_actual_scenario(scenario)
    actual_model = get_actual_model(model, scenario)

    if mode == 'accuracy':
        accuracy = get_accuracy(folder=f"{best_res_base_dir}/{BEST_RESULT_ACCURACY_SUB_FOLDER}")
        if not force and accuracy and accuracy.is_valid:
            print("Valid accuracy already exists. Skipping prepare.")
            return

        accuracy_folder = f"{best_res_base_dir}/{BEST_RESULT_ACCURACY_SUB_FOLDER}"
        harnessCfg = create_harness_config(config_path=best_res_base_dir,
                                        config_name=MODEL_CONF_NAME_WO_EXT,
                                        user_conf_path=best_res_base_dir,
                                        user_conf_name=USER_CONF_NAME,
                                        output_log_dir=accuracy_folder,
                                        test_mode="accuracy")
        assert harnessCfg.benchmark_name == actual_model, f"Parameter model and config benchmark_name are different: {actual_model} vs {harnessCfg.benchmark_name}"
        assert harnessCfg.scenario == actual_scenario.lower(), f"Parameter scenario and config scenario are different: {actual_scenario.lower()} vs {harnessCfg.scenario}"
        assert harnessCfg.test_mode == "accuracy", f"test_mode should be 'accuracy', actual is '{harnessCfg.test_mode}'"
        assert harnessCfg.backend == "vllm", f"backend should be 'vllm', actual is '{harnessCfg.backend}'"

        power_setting(actual_model, scenario)
        print("Running MLPerf accuracy...")
        run_mlperf_tests(harnessCfg)

        if model == MIXTRAL:
            print("Creating venv for mixtral")
            run_cmd(['bash', f'{CODE_DIR}/scripts/setup_mixtral_accuracy_env.sh'])
        model_name = model.split("-")[0] # ignore model size
        print("Running accuracy check...")
        result = run_cmd(['bash', f'{CODE_DIR}/scripts/check_{model_name}_accuracy_scores.sh', f'{accuracy_folder}/mlperf_log_accuracy.json'])
        assert "accuracy.txt for the accuracy scores" in result.stdout
        accuracy = get_accuracy(folder=f"{best_res_base_dir}/{BEST_RESULT_ACCURACY_SUB_FOLDER}")
        print(f"{scenario} Accuracy finished, it is {'valid' if accuracy and accuracy.is_valid else 'invalid'}")
    elif mode == 'compliance':
        verify_output_dir = f"{best_res_base_dir}/{BEST_RESULT_COMPLIANCE_SUB_FOLDER}"
        if not force and get_compliance(verify_output_dir).is_valid:
            print("Valid compliance already exists. Skipping prepare.")
            return

        run_output_dir = f"{verify_output_dir}/TEST06"
        audit_file = f"{TEST06_DIR}/{AUDIT_CONF}"
        shutil.copy(audit_file, '.')
        harnessCfg = create_harness_config(config_path=best_res_base_dir,
                                            config_name=MODEL_CONF_NAME_WO_EXT,
                                            user_conf_path=best_res_base_dir,
                                            user_conf_name=USER_CONF_NAME,
                                            output_log_dir=run_output_dir,
                                            test_mode="performance")
        power_setting(actual_model, scenario)
        print("Running MLPerf compliance...")
        run_mlperf_tests(harnessCfg)
        print("Running compliance verify check...")
        run_cmd(['python3', f'{TEST06_DIR}/run_verification.py', '-c', run_output_dir, '-o', verify_output_dir, '-s', actual_scenario])
        os.remove(AUDIT_CONF)
        print(f"{scenario} Compliance finished, it is {'valid' if get_compliance(verify_output_dir).is_valid else 'invalid'}")
    else:
        raise ValueError(f"Invalid prepare mode: {mode}")


def status(model):
    print(f"Running status with {model=}")
    for scenario in SUPPORTED_SCENARIOS:
        print(f'  {scenario=} PERF | {get_result(folder=f"{BEST_RESULTS}/{model}/{scenario}/{BEST_RESULT_PERFORMANCE_SUB_FOLDER}")}')
        print(f'  {scenario=} ACC  | {get_accuracy(folder=f"{BEST_RESULTS}/{model}/{scenario}/{BEST_RESULT_ACCURACY_SUB_FOLDER}")}')
        print(f'  {scenario=} COMP | {get_compliance(folder=f"{BEST_RESULTS}/{model}/{scenario}/{BEST_RESULT_COMPLIANCE_SUB_FOLDER}")}')


@debug
def get_result(folder):
    scenario = next(s for s in SUPPORTED_SCENARIOS if s in folder)
    assert scenario in SUPPORTED_SCENARIOS
    log_file = f"{folder}/mlperf_log_detail.txt"
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        lines = f.readlines()

    result = ExperimentResult()

    def get_value(data, key):
        return data.get('value', '') if data.get('key', '') == key else ''

    for line in lines:
        data = json.loads(line.strip(':::MLLOG '))
        if (value := get_value(data, "result_validity")) != '': result.is_valid = value == "VALID"
        if (value := get_value(data, "requested_scenario")) != '': result.requested_scenario = value
        if (value := get_value(data, "result_tokens_per_second")) != '': result.tokens_per_sec = float(value)
        if (value := get_value(data, "result_samples_per_second")) != '': result.samples_per_second = float(value)
        if (value := get_value(data, "result_completed_samples_per_sec")) != '': result.completed_samples_per_second = float(value)
        if (value := get_value(data, "result_completed_tokens_per_second")) != '': result.completed_tokens_per_second = float(value)
        if (value := get_value(data, "result_first_token_99.00_percentile_latency_ns")) != '': result.ttft = int(value)
        if (value := get_value(data, "result_time_per_output_token_99.00_percentile_ns")) != '': result.tpot = int(value)
    scenario = 'Server' if scenario == 'Interactive' else scenario
    assert result.requested_scenario == scenario
    return result


@debug
def get_accuracy(folder):
    scenario = next(s for s in SUPPORTED_SCENARIOS if s in folder)
    assert scenario in SUPPORTED_SCENARIOS
    accuracy_file = f"{folder}/accuracy.txt"
    if not os.path.exists(accuracy_file):
        return None
    with open(accuracy_file, 'r') as f:
        lines = f.readlines()
    accuracy = ExperimentAccuracy()
    for line in lines:
        if line.startswith('{') and line.endswith('}\n') and 'rouge1' in line:
            data = json.loads(line.replace('\'','"'))
            accuracy.rouge1 = float(data['rouge1'])
            accuracy.rouge2 = float(data['rouge2'])
            accuracy.rougeL = float(data['rougeL'])
            accuracy.tokens_per_sample = float(data['tokens_per_sample'])
            accuracy.gsm8k = float(data.get('gsm8k', 0))
            accuracy.mbxp = float(data.get('mbxp', 0))
    accuracy.requested_model = MIXTRAL if accuracy.gsm8k > 0 else LLAMA2

    accuracy.is_valid = all(accuracy.__dict__[metric] > limit for metric, limit in map_accuracy_limits[accuracy.requested_model].items()) \
                    and all(accuracy.__dict__[metric] < limit for metric, limit in map_accuracy_upper_limits[accuracy.requested_model].items())

    return accuracy

@debug
def get_compliance(folder):
    scenario = next(s for s in SUPPORTED_SCENARIOS if s in folder)
    assert scenario in SUPPORTED_SCENARIOS
    compliance_file = f"{folder}/TEST06/verify_accuracy.txt"
    verify_success = 'TEST06 verification complete'
    result = ExperimentCompliance()

    if os.path.exists(compliance_file):
        result.is_valid = verify_success in open(compliance_file).read()

    return result


def package(model_name):
    print(f"Running package with {model_name=}")
    from argparse import Namespace
    from package_submission import main as ps_main

    gpu_count = os.getenv('GPU_COUNT')
    gpu_name = os.getenv('GPU_NAME')
    cpu_count = os.getenv('CPU_COUNT')
    cpu_name = os.getenv('CPU_NAME')
    company = os.getenv('COMPANY')

    msg = "Environment variable {} is not set"
    assert gpu_count, msg.format("GPU_COUNT")
    assert gpu_name, msg.format("GPU_NAME")
    assert cpu_count, msg.format("CPU_COUNT")
    assert cpu_name, msg.format("CPU_NAME")
    assert company, msg.format("COMPANY")

    if os.path.exists(f'{CODE_DIR}/moe_accuracy_venv'):
        shutil.rmtree(f'{CODE_DIR}/moe_accuracy_venv')

    best_res_base_dir = f"{BEST_RESULTS}/{model_name}"
    user_conf_content = set()
    scenarios = []
    for scenario in SUPPORTED_SCENARIOS:
        if not os.path.exists(os.path.join(best_res_base_dir, scenario)):
            continue
        scenarios.append(scenario)
        # Overwrite model config
        shutil.copyfile(os.path.join(best_res_base_dir, scenario, MODEL_CONF_NAME),
                        os.path.join(CODE_DIR, "harness_llm", "models", model_name, f"{scenario.lower()}_{gpu_name}.yaml"))

        with open(os.path.join(best_res_base_dir, scenario, USER_CONF_NAME), 'r') as conf:
            for line in conf.readlines():
                if line.startswith(model_name):
                    user_conf_content.add(line)

    user_conf_path = os.path.join(CODE_DIR, f"user_{gpu_name}.conf")
    with open(user_conf_path, 'w') as conf:
        for line in user_conf_content:
            conf.write(line)

    benchmarks = [model_name]
    if model_name == LLAMA2:
        benchmarks = [f"{LLAMA2}-99", f"{LLAMA2}-99.9"]

    for benchmark in benchmarks:
        args = {
            "input_dir": f"{BEST_RESULTS}/{model_name}",
            "base_package_dir": SUBMISSION_PACKAGE_NAME,
            "scenarios": scenarios,
            "system_name": f'{gpu_count}x{gpu_name.upper()}_{cpu_count}x{cpu_name}',
            "benchmark": benchmark,
            "company": company,
            "user_conf": user_conf_path,
            "system_json": f'{gpu_name}_system.json',
            # defaults
            "division": "closed",
            "code_dir": CODE_DIR,
            "setup_dir": os.path.join(os.path.dirname(__file__), '..', 'setup'),
            "tools_dir": os.path.join(os.path.dirname(__file__), '..', 'tools'),
            "mlperf_inference_dir": "/app/mlperf_inference",
            "mlperf_inference_version": "v5.1",
        }

        ps_main(Namespace(**args))

    shutil.make_archive(SUBMISSION_PACKAGE_NAME, 'zip', SUBMISSION_PACKAGE_NAME)
    print(f"Done, check {SUBMISSION_PACKAGE_NAME}.zip")


def main():
    args = parse_arguments()
    if "scenario" in args and args.scenario == "Interactive" and args.model != LLAMA2:
        raise RuntimeError(f"Interactive only supported for {LLAMA2}")

    if args.command == 'experiment':
        experiment(args.model, args.scenario, args.model_conf, args.user_conf)
    elif args.command == 'update_best':
        update_best(args.model, args.scenario)
    elif args.command == 'prepare':
        prepare(args.model, args.mode, args.scenario, args.force)
    elif args.command == 'status':
        status(args.model)
    elif args.command == 'package':
        package(args.model)

if __name__ == '__main__':
    mp.set_start_method("spawn")
    main()
