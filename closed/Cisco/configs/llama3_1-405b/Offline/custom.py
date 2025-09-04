# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json
import os
import sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common.constants import Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.llama3_1-405b")
GPUBaseConfig = ParentConfig.GPUBaseConfig

class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    min_duration = 600000
    enable_sort = False

    trtllm_build_flags = {
    }

    trtllm_runtime_flags = {
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
    }


class HopperOfflineGPUBaseConfig(OfflineGPUBaseConfig):
    precision = "fp8"
    vboost_slider = 1

    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.95,
        'enable_chunked_context': True,
    }

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

    trtllm_build_flags = {
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'use_fp8_context_fmha': 'enable',
    }

# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json
import os
import sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common.constants import Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.llama3_1-405b")
GPUBaseConfig = ParentConfig.GPUBaseConfig



