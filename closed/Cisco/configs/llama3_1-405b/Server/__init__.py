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
from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.llama3_1-405b")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    min_duration = 600000
    enable_sort = False

    trtllm_build_flags = {
    }

    trtllm_runtime_flags = {
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
    }


class HopperServerGPUBaseConfig(ServerGPUBaseConfig):
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


class BlackwellServerGPUBaseConfig(ServerGPUBaseConfig):
    precision = 'fp4'
    # vboost_slider = 1

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
        'norm_quant_fusion': 'enable',
        'gemm_plugin': 'fp4',
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP8PP1")
class H100_SXM_80GB_TP4PP2(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx8

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


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP8PP1")
class H200_SXM_141GBx8_TP8PP1(HopperServerGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx8

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


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2PP2")
class B200_SXM_180GB_TP2PP2x1(BlackwellServerGPUBaseConfig):
    system = KnownSystem.B200_SXM_180GBx4

    server_target_qps = 0.65
    trtllm_build_flags = {
        'max_num_tokens': 1024,
        'tensor_parallelism': 2,
        'pipeline_parallelism': 2,
        # Disable to prevent intermittent failures; please only enable if you have collected a set of results.
        # 'gemm_allreduce_plugin': 'float16',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1024}
    gpu_batch_size = {'llama3.1-405b': 128}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2PP2")
class B200_SXM_180GB_TP2PP2x2(B200_SXM_180GB_TP2PP2x1):
    system = KnownSystem.B200_SXM_180GBx8
    server_target_qps = B200_SXM_180GB_TP2PP2x1.server_target_qps * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2PP2")
class GB200_NVL_186GB_ARM_TP2PP2x1(BlackwellServerGPUBaseConfig):
    system = KnownSystem.GB200_NVL_186GB_ARMx4

    server_target_qps = 0.75
    trtllm_build_flags = {
        'max_num_tokens': 1024,
        'tensor_parallelism': 2,
        'pipeline_parallelism': 2,
    }
    trtllm_runtime_flags = {'max_num_tokens': 1024}
    gpu_batch_size = {'llama3.1-405b': 128}


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2PP2DP1")
class GB200_NVL_186GB_ARM_TP2PP2x1_Triton(GB200_NVL_186GB_ARM_TP2PP2x1):
    use_triton = True
    triton_num_clients_per_frontend = 4
    triton_num_frontends_per_model = 2
    min_duration = 5_400_000

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2PP2DP4")
class GB200_NVL_186GB_ARM_TP2PP2x4_Triton(GB200_NVL_186GB_ARM_TP2PP2x1_Triton):
    server_target_qps = GB200_NVL_186GB_ARM_TP2PP2x1_Triton.server_target_qps * 4

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2PP2DP9")
class GB200_NVL_186GB_ARM_TP2PP2x9_Triton(GB200_NVL_186GB_ARM_TP2PP2x1_Triton):
    server_target_qps = GB200_NVL_186GB_ARM_TP2PP2x1_Triton.server_target_qps * 9

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2PP2DP18")
class GB200_NVL_186GB_ARM_TP2PP2x18_Triton(GB200_NVL_186GB_ARM_TP2PP2x1_Triton):
    server_target_qps = GB200_NVL_186GB_ARM_TP2PP2x1_Triton.server_target_qps * 18


