# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE100_RTX2000E_ADA_16GBX1(OfflineGPUBaseConfig):
    system = KnownSystem.SE100_RTX2000E_Ada_16GBx1

    workspace_size = 60000000000
    gpu_batch_size = {'clip1': 1 * 2, 'clip2': 1 * 2, 'unet': 1 * 2, 'vae': 1}
    offline_expected_qps = 0.15
    use_graphs = True
    numa_config = "0:0-5"

