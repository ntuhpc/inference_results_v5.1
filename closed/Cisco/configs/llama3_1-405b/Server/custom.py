# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class DGX_H200_H200_PCIE_80GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.DGX_H200_H200_PCIE_80GBX8

    gpu_batch_size = {'llama3.1-405b': 256}
    trtllm_build_flags = {
        'max_num_tokens': 8192,
        'tensor_parallelism': 8,
        'pipeline_parallelism': 1,
    }
    trtllm_runtime_flags = {
        'max_num_tokens': 2560,
        'max_batch_size': 64,
        'kvcache_free_gpu_mem_frac': 0.9
    }

    server_target_qps = 0.42


