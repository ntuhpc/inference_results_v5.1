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

GPTJComponent = import_module("code.gptj.tensorrt.constants").GPTJComponent


class GPTJ6BEngineBuilder(TRTLLMBuilder,
                          MLPerfInferenceEngine,
                          ArgDiscarder):
    """
    GPT-6B TRTLLM engine builder.
    """
    DEFAULT_CKPNT_PATH = Path("/work/build/models/GPTJ-6B")

    def __init__(
        self,
        model_name: str = "GPTJ-6B",
        model_path: os.PathLike = "/work/build/models/GPTJ-6B/checkpoint-final",
        checkpoint_dir: Optional[os.PathLike] = None,

        precision: str = "fp8",
        batch_size: int = 16,

        calib_dataset_path: os.PathLike = "/work/build/preprocessed_data/gptj/mlperf_gptj_openorca_calibration_1k/",
        calib_batch_size: int = 1024,

        *args,
        **kwargs,
    ):
        if checkpoint_dir is None:
            # NOTE(vir): enable int4_awq as needed for orin
            checkpoint_subdir = {'fp8': 'fp8-quantized-modelopt/GPTJ-FP8-quantized'}[precision]
            checkpoint_path = GPTJ6BEngineBuilder.DEFAULT_CKPNT_PATH / checkpoint_subdir
        else:
            checkpoint_path = Path(checkpoint_dir)

        assert checkpoint_path.parent.exists(), f"Checkpoint parent directory does not exist: {checkpoint_path.parent}"
        assert precision in ['fp8'], f"Unsupported Precision for GPTJ-6B: {precision}"

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


class GPTJ6BEngineBuilderOp(Operation,
                            ArgDiscarder):
    COMPONENT_BUILDER_MAP = {
        GPTJComponent.GPTJ: GPTJ6BEngineBuilder,
    }

    @classmethod
    def immediate_dependencies(cls):
        # TODO: Integrate dataset scripts as deps
        return None

    def __init__(self,
                 *args,
                 batch_size: Dict[GPTJComponent, int] = None,
                 config_ver: str = "default",
                 **kwargs):
        """Creates a GPTJ6BEngineBuilderOp.

        Args:
            batch_size (Dict[str, int]): Component and its batch size to build the engine for)
            config_ver (str): Legacy field. Identifier for the benchmark configuration. (Default: "default").
        """
        super().__init__(*args, **kwargs)
        self.engine_dir = dict_get(kwargs, "engine_dir", default=None)
        self.config_ver = config_ver

        if not batch_size:
            logging.warning("No batch_size dict provided for GPTJ6BEngineBuilderOp. Setting to default value {GPTJComponent.GPTJ : 1}")
            batch_size = {GPTJComponent.GPTJ: 1}

        self.builders = []
        for component, component_batch_size in batch_size.items():
            builder = GPTJ6BEngineBuilderOp.COMPONENT_BUILDER_MAP[component](*args, batch_size=component_batch_size, **kwargs)
            self.builders.append(builder)

    def run(self, scratch_space, _):
        for builder in self.builders:
            if self.engine_dir is not None:
                engine_dir = Path(self.engine_dir)
            else:
                engine_dir = builder.engine_dir(scratch_space)
                engine_dir = engine_dir / f"bs{builder.max_batch_size}-{self.config_ver}"

            # start build
            builder(engine_dir)


class GPTJ6B(LegacyBuilder):
    """Temporary spoofing class to wrap around Mitten to adhere to the old API.
    """

    def __init__(self, args):
        super().__init__(GPTJ6BEngineBuilderOp(**args))
