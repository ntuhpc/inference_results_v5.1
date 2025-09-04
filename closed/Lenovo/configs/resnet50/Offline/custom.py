# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE100_RTX2000E_ADA_16GBX1(OfflineGPUBaseConfig):
    system = KnownSystem.SE100_RTX2000E_Ada_16GBx1
    gpu_batch_size = {'resnet50': 32}
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 6900 
    use_graphs = True
    numa_config = "0:0-5"

