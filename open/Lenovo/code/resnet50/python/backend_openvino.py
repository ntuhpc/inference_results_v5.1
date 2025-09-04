"""
openvino backend (https://docs.openvino.ai/)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np
from openvino import Core, get_version, CompiledModel, Output
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import backend

class BackendOpenvino(backend.Backend):
    def __init__(self) -> None:
        super().__init__()
        self.core: Core | None = None
        self.compiled_model: CompiledModel | None = None
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.output_ports: Dict[str, Output] = {}

    def version(self) -> str:
        return get_version()

    def name(self) -> str:
        """Name of the runtime."""
        return "openvino"

    def image_format(self) -> str:
        """image_format. Please use --data-format=NHWC for alternative layout."""
        return "NCHW"

    def load(
        self,
        model_path: str,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        ) -> "BackendOpenvino":

        """Load model and find input/outputs from the model file."""
        self.core = self.core or Core()


        device = os.environ.get("OPENVINO_DEVICE", "CPU")
        print("[Info] Running Inference on {}".format(device))
        
        self.core.set_property(device, {
            hints.performance_mode: hints.PerformanceMode.LATENCY,
        })

        model = self.core.read_model(model_path)
        model.reshape([1,3,224,224])

        if device.startswith("NPU"):
            batch = 1
            shape_map = {}
            for port in model.inputs:
                orig_shape = list(port.partial_shape)
                if len(orig_shape) < 4:
                    continue
                orig_shape[0] = batch
                shape_map[port] = orig_shape
            if shape_map:
                model.reshape(shape_map)

        self.compiled_model = self.core.compile_model(
            model,
            device,
            {
                hints.num_requests: "1",
                props.streams.num: "1",
                props.enable_profiling: False,
                hints.performance_mode: hints.PerformanceMode.LATENCY,
            }
        )

        self.req = self.compiled_model.create_infer_request()

        self.inputs = list(inputs) if inputs else [p.get_any_name() for p in self.compiled_model.inputs]
        self.outputs = list(outputs) if outputs else [p.get_any_name() for p in self.compiled_model.outputs]

        self._output_ports = {p.get_any_name(): p for p in self.compiled_model.outputs}
        return self

    def predict(self, feed: Dict[str, np.ndarray]):
        """Run the prediction."""

        if self.compiled_model is None:
            raise RuntimeError("Model has not been loaded - call .load() first.")

        for name, arr in feed.items():
            self.req.get_tensor(name).data[...] = arr
        self.req.infer()

        output = self.req.get_output_tensor(0).data
        return [output]
