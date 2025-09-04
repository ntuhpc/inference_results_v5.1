from omegaconf import OmegaConf
import multiprocessing as mp
import time
import psutil
import subprocess
import os
from pathlib import Path
import mlperf_loadgen as lg
from harness_llm.backends.vllm.sut_offline import SUTvLLMOffline
from harness_llm.backends.vllm.sync_sut import SyncServerSUT
from harness_llm.backends.vllm.async_sut import AsyncServerSUT
from harness_llm.loadgen.sut import SUT
from harness_llm.common.config import hydra_runner, Config, HarnessConfig
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}

sample_count_map = {
    "llama2-70b": 24576,
    "llama2-70b-interactive": 24576,
    "mixtral-8x7b": 15000,
    "llama3_1-405b": 8313,
}

def get_mlperf_test_settings(
    benchmark: str, scenario: str, test_mode: str, harness_config: HarnessConfig
) -> lg.TestSettings:
    "Returns the test settings needed for mlperf"
    settings = lg.TestSettings()
    settings.scenario = scenario_map[scenario.lower()]
    # settings.FromConfig(harness_config.mlperf_conf_path, benchmark.lower(), scenario)
    settings.FromConfig(harness_config.user_conf_path, benchmark.lower(), scenario)

    if harness_config.target_qps > 0:
        settings.offline_expected_qps = harness_config.target_qps
        settings.server_target_qps = harness_config.target_qps
        log.warning(
            f"Overriding default QPS with {harness_config.target_qps}"
        )

    if harness_config.total_sample_count != sample_count_map[benchmark.lower()]:
        settings.min_query_count = harness_config.total_sample_count
        settings.max_query_count = harness_config.total_sample_count
        log.warning(
            f"Overriding default sample count with {harness_config.total_sample_count}"
        )

    if harness_config.duration_sec != -1:
        time_ms = harness_config.duration_sec * 1000
        settings.min_duration_ms = time_ms
        settings.max_duration_ms = time_ms
        log.warning(
            f"Overriding default duration with {time_ms} ms"
        )

    if test_mode.lower() == "accuracy":
        settings.mode = lg.TestMode.AccuracyOnly
        log.warning(
            "Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet"
        )
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    return settings


def get_mlperf_log_settings(harness_config: HarnessConfig) -> lg.LogOutputSettings:
    # create the log dir if not exist.
    os.makedirs(harness_config.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = harness_config.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = harness_config.enable_log_trace
    return log_settings


def get_sut(scenario: str, server_version: str, conf: dict) -> SUT:
    """
    This returns an instance of SUT depends on the inputs.
    """
    if scenario.lower() == "offline":
        return SUTvLLMOffline(conf)
    elif scenario.lower() == "server":
        if server_version.lower() == "sync":
            return SyncServerSUT(conf)
        elif server_version.lower() == "async":
            return AsyncServerSUT(conf)
        else:
            raise ValueError(f"Unsupported server version is passed - {server_version}")
    else:
        raise ValueError(f"Unsupported scenario is passed - {scenario}")


def set_mlperf_envs(env_config: dict):
    print(f"{env_config=}", flush=True)
    for env, val in env_config.items():
        if val is not None:
            os.environ[env] = str(val)
            log.info(f"Setting {env} to {val}")


def run_mlperf_tests(cfg: Config) -> None:
    """
    A main entry point to run the mlperf tests.
    """

    # Set mlperf test and log settings
    test_settings = get_mlperf_test_settings(
        benchmark=cfg.benchmark_name,
        scenario=cfg.scenario,
        test_mode=cfg.test_mode,
        harness_config=cfg.harness_config,
    )

    log_settings = get_mlperf_log_settings(harness_config=cfg.harness_config)

    conf = OmegaConf.to_container(cfg, throw_on_missing=True)

    # Set ENVs for mlperf
    set_mlperf_envs(conf["env_config"])

    # Instantiate SUT
    sut = get_sut(scenario=cfg.scenario, server_version=cfg.server_version, conf=conf)

    log.info("Instantiating SUT")
    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    print(OmegaConf.to_yaml(cfg))
    lg.StartTestWithLogSettings(lgSUT, sut.qsl, test_settings, log_settings)
    log.info("Completed benchmark run")
    sut.stop()
    log.info("Run Completed!")
    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)
    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


@hydra_runner(schema=Config)
def run_from_cli(cfg: Config) -> None:
    run_mlperf_tests(cfg)


def run_from_optuna(config_path, config_name, overrides) -> None:

    @hydra_runner(config_path=config_path, config_name=config_name, config_overrides=overrides)
    def _run_mlperf_tests(cfg: Config) -> None:
        try:
            run_mlperf_tests(cfg)
        except:
            logging.exception("Something went wrong")

    _run_mlperf_tests()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    log.info(f"mp.get_context:{mp.get_context()}")
    run_from_cli()
