from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union
import os
import hydra
from hydra.core.config_store import ConfigStore
import argparse
import functools
import os
import sys
from typing import Any, Callable, Optional

from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.core.config_store import ConfigStore
from hydra.types import TaskFunction
from omegaconf import DictConfig, OmegaConf, MISSING


@dataclass
class EnvConfig:
   HIP_FORCE_DEV_KERNARG: int = 1
   VLLM_USE_TRITON_FLASH_ATTN: int = 0
   VLLM_FP8_PADDING: int = 1
   VLLM_FP8_ACT_PADDING: int = 1
   VLLM_FP8_WEIGHT_PADDING: int = 1
   VLLM_FP8_REDUCE_CONV: int = 1
   VLLM_SCHED_PREFILL_KVC_FREEPCT: float = 30.0
   HARNESS_DISABLE_VLLM_LOGS: int = 1
   VLLM_LOGGING_LEVEL: str = "ERROR"
   NCCL_MIN_NCHANNELS: Optional[int] = None
   RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES: Optional[int] = None
   TOKENIZERS_PARALLELISM: Optional[bool] = None
   VLLM_INSTALL_PUNICA_KERNELS: Optional[int] = None
   VLLM_USE_ROCM_CUSTOM_PAGED_ATTN: Optional[int] = None
   HARNESS_GC_LIMIT: Optional[int] = 100000
   VLLM_MOE_MLPERF_KERNEL: Optional[int] = 0
   VLLM_MOE_MLPERF_SCHED: Optional[int] = 0
   VLLM_MOE_MLPERF_SCHED_PREFILL_KVC_FREE_PCT: Optional[float] = 0.0
   VLLM_MOE_MLPERF_SCHED_PREFILL_KVC_TOKENS_PCT: Optional[float] = 0.0
   VLLM_MOE_MLPERF_SCHED_PREFILL_KVC_SEQS_PCT: Optional[float] = 0.0
   VLLM_SCHEDULER_SLOAWARE_ENABLE: Optional[int] = 0
   VLLM_SCHEDULER_SLOAWARE_TPOT_TARGET: Optional[float] = 200E-3
   VLLM_LLAMA2_MLPERF_SCHED: Optional[int] = 0
   VLLM_LLAMA2_MLPERF_MAX_TARGET_DECODE_BATCH: Optional[int] = 1000000
   VLLM_LLAMA2_MLPERF_MIN_TARGET_DECODE_BATCH: Optional[int] = 1000000
   VLLM_LLAMA2_MLPERF_STEP_DECODE_BATCH: Optional[int] = 0
   VLLM_LLAMA2_MLPERF_MIN_REQUIRE_PREFILL_GPU_BLOCK: Optional[int] = 0
   VLLM_LLAMA2_MLPERF_MIN_REQUIRE_PREFILL_QUERY: Optional[int] = 0
   HIPBLASLT_TUNING_OVERRIDE_FILE: Optional[str] = None



@dataclass
class HarnessConfig:
    target_qps: float = 80.0
    total_sample_count: int = 15000
    duration_sec: int = -1
    enable_log_trace: bool = False
    warmup_duration: float = 0.0
    enable_warm_up: bool = False
    sort_samples: Optional[str] = "ignore"
    schedule_algo: str = "shortest_queue"
    warm_up_sample_count_per_server: int = 50
    enable_batcher: bool = False
    batcher_threshold: float = 0.1
    gpu_batch_size: int = 128
    data_parallel_size: int = 1
    dataset_path: str = MISSING
    mlperf_conf_path: str = MISSING
    user_conf_path: str = MISSING
    output_log_dir: str = MISSING
    load_balance_token_weight: Optional[float] = 0.02
    load_balance_window_size: Optional[int] = 10


# Defaults for VLLM SamplingParams
@dataclass
class SamplingParamsInput:
    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    seed: Optional[int] = None
    top_p: float = 1
    top_k: int = 1
    min_p: float = 0
    ppl_measurement: bool = False
    future_context: Optional[List[int]] = None
    max_tokens: Optional[int] = 1024
    min_tokens: int = 1
    ignore_eos: bool = False
    detokenize: bool = False
    stop: Optional[List[str]] = None
    stop_seq_ids_config: Optional[Dict] = None


@dataclass
class LLMConfig:
    model: str = MISSING

    dtype: str = "float16"
    kv_cache_dtype: str = "fp8"
    quantization: Optional[str] = None

    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1

    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    swap_space: float = 4  # GiB
    block_size: int = 16

    num_scheduler_steps: int = 1

    enforce_eager: Optional[bool] = False
    max_seq_len_to_capture: int = 8192

    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256

    disable_custom_all_reduce: bool = False
    disable_log_stats: bool = True
    enable_chunked_prefill: Optional[bool] = None
    enable_prefix_caching: bool = False

@dataclass
class Config:
    benchmark_name: str = MISSING
    scenario: str = MISSING
    test_mode: str = "performance"
    server_version: str = "sync"
    # Arguments for vLLM engine.
    llm_config: LLMConfig = MISSING
    # Sampling parameters for text generation
    sampling_params: SamplingParamsInput = MISSING
    # Parameters for harness
    harness_config: HarnessConfig = MISSING
    # Env vars
    env_config: EnvConfig = MISSING


def hydra_runner(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    config_overrides: Optional[list[str]] = None,
    schema: Optional[Any] = Config,
) -> Callable[[TaskFunction], Any]:
    """
    Decorator used for passing the Config paths to main function.
    Optionally registers a schema used for validation/providing default values.

    Args:
        config_path: Optional path that will be added to config search directory.
        config_name: Pathname of the config file.
        schema: A schema
    """

    def decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def wrapper(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            # Check it config was passed.
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args = get_args_parser()

                # Parse arguments in order to retrieve overrides
                parsed_args = args.parse_args() if config_overrides is None else args.parse_args([])  # type: argparse.Namespace

                # Get overriding args in dot string format
                overrides = parsed_args.overrides  # type: list

                if config_overrides is not None:
                    overrides.extend(config_overrides)

                # Disable the creation of .hydra subdir
                # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory
                overrides.append("hydra.output_subdir=null")
                # Hydra logging outputs only to stdout (no log file).
                # https://hydra.cc/docs/configure_hydra/logging
                overrides.append("hydra/job_logging=stdout")
                overrides.append("hydra.job.chdir=False")

                # Wrap a callable object with name `parse_args`
                # This is to mimic the ArgParser.parse_args() API.
                def parse_args(self, args=None, namespace=None):
                    return parsed_args

                parsed_args.parse_args = parse_args

                # no return value from run_hydra() as it may sometime actually run the task_function
                # multiple times (--multirun)
                # argparse_wrapper = _argparse_wrapper(args)
                argparse_wrapper = parsed_args

                conf_path = (
                    config_path if config_path is not None else parsed_args.config_path
                )
                conf_name = (
                    config_name if config_name is not None else parsed_args.config_name
                )

                if conf_path is None or conf_name is None:
                    sys.stderr.write(
                        f"Missing --config-path and --config-name required params!\n"
                    )
                    sys.exit(1)

                # # Create config store.
                cs = ConfigStore.instance()
                # Get the correct ConfigStore "path name" to "inject" the schema.
                path, name = os.path.split(conf_name)
                # Make sure the path is not set - as this will disable validation scheme.
                if path != "":
                    sys.stderr.write(
                        f"ERROR Cannot set config file path using `--config-name` when "
                        "using schema. Please set path using `--config-path` and file name using "
                        "`--config-name` separately.\n"
                    )
                    sys.exit(1)
                else:
                    name = conf_name

                # Register the configuration as a node under the name in the group.
                cs.store(name=name, node=schema)  # group=group,

                _run_hydra(
                    args=argparse_wrapper,
                    args_parser=args,
                    task_function=task_function,
                    config_path=conf_path,
                    config_name=conf_name,
                )

        return wrapper

    return decorator
