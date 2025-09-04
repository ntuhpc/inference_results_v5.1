# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common.constants import Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.llama2-70b")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    min_duration = 2400000
    enable_sort = False


class HopperServerGPUBaseConfig(ServerGPUBaseConfig):
    precision = "fp8"
    vboost_slider = 1

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }


class BlackwellServerGPUBaseConfig(ServerGPUBaseConfig):
    precision = 'fp4'

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
        'norm_quant_fusion': 'enable'
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_144GB_aarch64x1(HopperServerGPUBaseConfig):
    system = KnownSystem.GH200_144GB_ARMx1

    gpu_batch_size = {'llama2-70b': 2048}
    server_target_qps = 15
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_144GB_aarch64x1_HighAccuracy(GH200_144GB_aarch64x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_144GB_aarch64x2(GH200_144GB_aarch64x1):
    system = KnownSystem.GH200_144GB_ARMx2
    server_target_qps = GH200_144GB_aarch64x1.server_target_qps * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_144GB_aarch64x2_HighAccuracy(GH200_144GB_aarch64x2):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "PP2")
class H100_SXM_80GB_PP2x1(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx2
    vboost_slider = 0
    min_duration = 3600000

    gpu_batch_size = {'llama2-70b': 1024}
    server_target_qps = 25.25
    trtllm_build_flags = {
        'max_num_tokens': 1024,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 2,
        'reduce_fusion': 'enable',
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {
        'max_num_tokens': 1024,
        'kvcache_free_gpu_mem_frac': 0.95,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "PP2")
class H100_SXM_80GB_PP2x4(H100_SXM_80GB_PP2x1):
    system = KnownSystem.H100_SXM_80GBx8
    server_target_qps = 107


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "PP2")
class H100_SXM_80GB_HighAccuracy_PP2x1(H100_SXM_80GB_PP2x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "PP2")
class H100_SXM_80GB_HighAccuracy_PP2x4(H100_SXM_80GB_PP2x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class H100_SXM_80GB_TP2x1(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx2

    gpu_batch_size = {'llama2-70b': 2048}
    server_target_qps = 13.533

    trtllm_build_flags = {
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
        'gemm_swiglu_plugin': 'fp8',
    }


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class H100_SXM_80GB_Triton_TP2x1(H100_SXM_80GB_TP2x1):
    use_triton = True
    triton_num_clients_per_frontend = 2
    triton_num_frontends_per_model = 4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class H100_SXM_80GB_TP2x2(H100_SXM_80GB_TP2x1):
    system = KnownSystem.H100_SXM_80GBx4
    server_target_qps = 18.4 * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class H100_SXM_80GB_TP2x4(H100_SXM_80GB_TP2x2):
    system = KnownSystem.H100_SXM_80GBx8
    server_target_qps = 75


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ, "TP2")
class H100_SXM_80GB_MaxQ_TP2x4(H100_SXM_80GB_TP2x4):
    system = KnownSystem.H100_SXM_80GBx8
    server_target_qps = 13.5 * 4
    power_limit = 450


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "TP2")
class H100_SXM_80GB_HighAccuracy_TP2x1(H100_SXM_80GB_TP2x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "TP2")
class H100_SXM_80GB_HighAccuracy_TP2x4(H100_SXM_80GB_TP2x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ, "TP2")
class H100_SXM_80GB_HighAccuracy_MaxQ_TP2x4(H100_SXM_80GB_MaxQ_TP2x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class H100_NVL_94GB_TP2x1(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx2

    gpu_batch_size = {'llama2-70b': 640}
    server_target_qps = 12.5

    trtllm_build_flags = {
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class H100_NVL_94GB_TP2x2(H100_NVL_94GB_TP2x1):
    system = KnownSystem.H100_NVL_94GBx4
    server_target_qps = 12.5 * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class H100_NVL_94GB_TP2x4(H100_NVL_94GB_TP2x2):
    system = KnownSystem.H100_NVL_94GBx8
    server_target_qps = 12.5 * 4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ, "TP2")
class H100_NVL_94GB_MaxQ_TP2x4(H100_NVL_94GB_TP2x4):
    server_target_qps = 38
    power_limit = 350


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "TP2")
class H100_NVL_94GB_HighAccuracy_TP2x1(H100_NVL_94GB_TP2x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "TP2")
class H100_NVL_94GB_HighAccuracy_TP2x4(H100_NVL_94GB_TP2x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ, "TP2")
class H100_NVL_94GB_HighAccuracy_MaxQ_TP2x4(H100_NVL_94GB_MaxQ_TP2x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(HopperServerGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1
    gpu_batch_size = {'llama2-70b': 2048}
    server_target_qps = 14.28
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H200_SXM_141GBx1_MaxQ(H200_SXM_141GBx1):
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx1_HighAccuracy(H200_SXM_141GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H200_SXM_141GBx1_HighAccuracy_MaxQ(H200_SXM_141GBx1_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1_Triton(H200_SXM_141GBx1):
    use_triton = True
    triton_num_clients_per_frontend = 2
    triton_num_frontends_per_model = 4
    server_target_qps = 12.5


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx1_HighAccuracy_Triton(H200_SXM_141GBx1_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx8
    server_target_qps = H200_SXM_141GBx1.server_target_qps * 8


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8_Triton(H200_SXM_141GBx8):
    use_triton = True
    triton_num_clients_per_frontend = 2
    triton_num_frontends_per_model = 4
    triton_num_servers = 8


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx8_HighAccuracy_Triton(H200_SXM_141GBx8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H200_SXM_141GBx8_MaxQ(H200_SXM_141GBx1_MaxQ):
    power_limit = 500
    system = KnownSystem.H200_SXM_141GBx8
    server_target_qps = 80


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx8_HighAccuracy(H200_SXM_141GBx8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H200_SXM_141GBx8_HighAccuracy_MaxQ(H200_SXM_141GBx8_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx1(HopperServerGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GB_CTSx1
    gpu_batch_size = {'llama2-70b': 850}
    server_target_qps = 14.5


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx1_HighAccuracy(H200_SXM_141GB_CTSx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx8(H200_SXM_141GB_CTSx1):
    system = KnownSystem.H200_SXM_141GB_CTSx8
    gpu_batch_size = {'llama2-70b': 1024}
    server_target_qps = 113.5


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx8_HighAccuracy(H200_SXM_141GB_CTSx8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class B200_SXM_180GBx1(BlackwellServerGPUBaseConfig):
    system = KnownSystem.B200_SXM_180GBx1
    vboost_slider = 1
    server_target_qps = 45.5
    gpu_batch_size = {'llama2-70b': 2048}
    trtllm_build_flags = {'max_num_tokens': 3584}
    trtllm_runtime_flags = {
        'max_num_tokens': 3584,
        'kvcache_free_gpu_mem_frac': 0.95,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class B200_SXM_180GBx1_HighAccuracy(B200_SXM_180GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class B200_SXM_180GBx8(B200_SXM_180GBx1):
    system = KnownSystem.B200_SXM_180GBx8
    server_target_qps = B200_SXM_180GBx1.server_target_qps * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class B200_SXM_180GBx8_HighAccuracy(B200_SXM_180GBx8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class DGX_H200_H200_PCIE_80GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.DGX_H200_H200_PCIE_80GBX8
    gpu_batch_size = {'llama2-70b': 2048}
    server_target_qps = 14.28*8
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DGX_H200_H200_PCIE_80GBx8_HighAccuracy(DGX_H200_H200_PCIE_80GBX8):
    pass

