# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE100_RTX2000E_ADA_16GBX1(ServerGPUBaseConfig):
    system = KnownSystem.SE100_RTX2000E_Ada_16GBx1

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    cache_file: str = ''
    engine_dir: str = ''
    schedule_rng_seed: int = 0
    sdxl_batcher_time_limit: float = 0.0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: float = 0.0
    server_target_qps_adj_factor: float = 0.0
    vboost_slider: int = 0
    workspace_size: int = 0


