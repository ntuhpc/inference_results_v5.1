# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE100_RTX2000E_ADA_16GBX1(OfflineGPUBaseConfig):
    system = KnownSystem.SE100_RTX2000E_Ada_16GBx1
    gpu_batch_size = {'3d-unet': 4}
    offline_expected_qps = 0.7 
    slice_overlap_patch_kernel_cg_impl = True
    numa_config = "0:0-5"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SE100_RTX2000E_ADA_16GBX1_HighAccuracy(SE100_RTX2000E_ADA_16GBX1):
    pass


