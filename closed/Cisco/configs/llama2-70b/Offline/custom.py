# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class DGX_H200_H200_PCIE_80GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.DGX_H200_H200_PCIE_80GBX8
    gpu_batch_size = {'llama2-70b': 2048}
    offline_expected_qps = 14.4*8
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DGX_H200_H200_PCIE_80GBX8_HighAccuracy(DGX_H200_H200_PCIE_80GBX8):
    pass


