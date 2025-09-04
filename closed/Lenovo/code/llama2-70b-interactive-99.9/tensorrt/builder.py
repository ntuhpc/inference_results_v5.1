#!/usr/bin/env python3
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

from __future__ import annotations
from importlib import import_module
import os
from pathlib import Path
from typing import Dict, Optional

from nvmitten.nvidia.builder import LegacyBuilder, MLPerfInferenceEngine
from nvmitten.pipeline import Operation
from nvmitten.utils import logging

from code.common import dict_get
from code.common.mitten_compat import ArgDiscarder
from code.common.llm_builder import TRTLLMBuilder

LLAMA2Component = import_module("code.llama2-70b.tensorrt.constants").LLAMA2Component


class LLAMA2EngineBuilder(TRTLLMBuilder,
                          MLPerfInferenceEngine,
                          ArgDiscarder):
    """
    Llama2-70B TRTLLM engine builder.
    """
    DEFAULT_CKPNT_PATH = Path("/work/build/models/Llama2")

    def __init__(
        self,
        model_name: str = "llama2-70b-chat-hf",
        model_path: os.PathLike = "/work/build/models/Llama2/Llama-2-70b-chat-hf",
        checkpoint_dir: Optional[os.PathLike] = None,

        precision: str = "fp8",
        batch_size: int = 16,

        calib_dataset_path: os.PathLike = "/work/build/preprocessed_data/llama2-70b/mlperf_llama2_openorca_calibration_1k",
        calib_batch_size: int = 1024,

        *args,
        **kwargs,
    ):
        if checkpoint_dir is None:
            checkpoint_subdir = {'fp8': 'fp8-quantized-modelopt', 'fp4': 'fp4-quantized-modelopt'}[precision]
            checkpoint_name = f'{model_name}-tp{kwargs["trtllm_build_flags"]["tensor_parallelism"]}pp{kwargs["trtllm_build_flags"]["pipeline_parallelism"]}-{precision}'
            checkpoint_path = LLAMA2EngineBuilder.DEFAULT_CKPNT_PATH / checkpoint_subdir / checkpoint_name
        else:
            checkpoint_path = Path(checkpoint_dir)

        assert checkpoint_path.parent.exists(), f"Checkpoint parent directory does not exist: {checkpoint_path.parent}"
        assert precision in ['fp8', 'fp4'], f"Unsupported Precision for Llama2-70B: {precision}"

        super().__init__(
            model_name=model_name,
            model_path=model_path,
            checkpoint_path=checkpoint_path,

            precision=precision,
            max_batch_size=batch_size,

            calib_dataset_path=calib_dataset_path,
            calib_batch_size=calib_batch_size,

            *args,
            **kwargs
        )


class LLAMA2EngineBuilderOp(Operation,
                            ArgDiscarder):
    COMPONENT_BUILDER_MAP = {
        LLAMA2Component.LLAMA2: LLAMA2EngineBuilder,
        LLAMA2Component.LLAMA2_Interactive: LLAMA2EngineBuilder,
    }

    @classmethod
    def immediate_dependencies(cls):
        # TODO: Integrate dataset scripts as deps
        return None

    def __init__(self,
                 *args,
                 batch_size: Dict[LLAMA2Component, int] = None,
                 workload_setting: Optional[WorkloadSetting] = None,
                 device_type: str = 'gpu',
                 **kwargs):
        """Creates a LLAMA2EngineBuilderOp.

        Args:
            batch_size (Dict[str, int]): Component and their batch sizes for engine-build.
            workload_setting (WorkloadSetting): Identifier for the benchmark configuration.
            device_type (str): Class of device this workload is running on. Not used for LLMs.
        """
        super().__init__(*args, **kwargs)
        self.engine_dir = dict_get(kwargs, "engine_dir", default=None)
        self.workload_setting = workload_setting
        self.device_type = device_type

        if not batch_size:
            logging.warning("No batch_size dict provided for LLAMA2EngineBuilderOp. Setting to default value {LLAMA2Component.LLAMA2 : 1}")
            batch_size = {LLAMA2Component.LLAMA2: 1}

        self.builders = []
        for component, component_batch_size in batch_size.items():
            builder = LLAMA2EngineBuilderOp.COMPONENT_BUILDER_MAP[component](*args, batch_size=component_batch_size, **kwargs)
            self.builders.append(builder)

    def run(self, scratch_space, dependency_outputs):
        for builder in self.builders:
            if self.engine_dir is not None:
                engine_dir = Path(self.engine_dir)

            else:
                engine_dir = builder.engine_dir(scratch_space)

                tag = f"tp{builder.tp_size}pp{builder.pp_size}.{self.workload_setting.shortname()}"
                engine_name = builder.engine_name(self.device_type,
                                                  builder.max_batch_size,
                                                  builder.precision,
                                                  tag=tag)
                engine_name = str(Path(engine_name).with_suffix("")).replace(".", '-')
                engine_dir = engine_dir / engine_name

            # start build
            builder(engine_dir)


class LLAMA2(LegacyBuilder):
    """
    Temporary spoofing class to wrap around Mitten to adhere to the old API.
    """

    def __init__(self, args):
        super().__init__(LLAMA2EngineBuilderOp(**args))
