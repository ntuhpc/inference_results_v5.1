#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch


from . import backend
from .dlrm_model import DLRMMLPerf

DEFAULT_INT_NAMES = [
    "int_0",
    "int_1",
    "int_2",
    "int_3",
    "int_4",
    "int_5",
    "int_6",
    "int_7",
    "int_8",
    "int_9",
    "int_10",
    "int_11",
    "int_12",
]


class BackendPytorchNative(backend.Backend):
    def __init__(
        self,
        num_embeddings_per_feature,
        embedding_dim=128,
        dcn_num_layers=3,
        dcn_low_rank_dim=512,
        dense_arch_layer_sizes=[512, 256, 128],
        over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
    ):
        super(BackendPytorchNative, self).__init__()

        self.model = None

        self.embedding_dim = embedding_dim
        self.num_embeddings_per_feature = num_embeddings_per_feature
        self.dcn_num_layers = dcn_num_layers
        self.dcn_low_rank_dim = dcn_low_rank_dim
        self.dense_arch_layer_sizes = dense_arch_layer_sizes
        self.over_arch_layer_sizes = over_arch_layer_sizes

        print("Using CPU...")

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native-dlrm"

    def load(self, args):

        # print(f"Loading model from {args.model_path}")
        print("Initializing model...")
        model = DLRMMLPerf(
            embedding_dim=self.embedding_dim,
            num_embeddings_pool=self.num_embeddings_per_feature,
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=self.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.over_arch_layer_sizes,
            dcn_num_layers=self.dcn_num_layers,
            dcn_low_rank_dim=self.dcn_low_rank_dim,
        )

        if args.use_int8:
            if args.calibration:
                print(f"Loading model fp32 weights for calibration: {args.model_path}")
                model.load_state_dict(torch.load(args.model_path))
                self.model = model
            else:
                del model
                print(f"Loading model int8 weights: {args.int8_model_path}")
                self.model = torch.jit.load(args.int8_model_path)
                print("int8 model ready")
        else:
            print(f"Loading model fp32 weights: {args.model_path}")
            model.load_state_dict(torch.load(args.model_path))
            self.model = model
            print("fp32 model ready...")
        if not args.calibration:
            self.model = self.model.cpu().share_memory()
            print("share_memory ready")
        self.model = model
        return self

    def batch_predict(self, densex, index, offset):
        with torch.no_grad():
            out = self.model(densex, index, offset)
            return out


def get_backend(backend, dataset):
    if backend == "pytorch-native":
        if dataset == "multihot-criteo":
            backend = BackendPytorchNative(
                num_embeddings_per_feature=[
                    40000000,
                    39060,
                    17295,
                    7424,
                    20265,
                    3,
                    7122,
                    1543,
                    63,
                    40000000,
                    3067956,
                    405282,
                    10,
                    2209,
                    11938,
                    155,
                    4,
                    976,
                    14,
                    40000000,
                    40000000,
                    40000000,
                    590152,
                    12973,
                    108,
                    36,
                ],
                embedding_dim=128,
                dcn_num_layers=3,
                dcn_low_rank_dim=512,
                dense_arch_layer_sizes=[512, 256, 128],
                over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
            )
        else:
            raise ValueError("only multihot-criteo dataset options are supported")

    else:
        raise ValueError("unknown backend: " + backend)
    return backend
