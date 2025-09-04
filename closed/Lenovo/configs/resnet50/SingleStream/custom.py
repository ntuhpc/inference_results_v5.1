# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE100_RTX2000E_ADA_16GBX1(SingleStreamGPUBaseConfig):
    system = KnownSystem.SE100_RTX2000E_Ada_16GBx1
    single_stream_expected_latency_ns = 530000 
    disable_beta1_smallk = True
    numa_config = "0:0-5"
