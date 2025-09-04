#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse


import numpy as np
import torch

import multihot_criteo
from backend_pytorch_native import get_backend
import torch.quantization
import sklearn

from torch.ao.quantization import (
    PerChannelMinMaxObserver,
    QConfig,
    MinMaxObserver
)
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import float_qparams_weight_only_qconfig_4bit, QConfigMapping


# pylint: disable=missing-docstring

# the datasets we support
SUPPORTED_DATASETS = {
    "multihot-criteo": (
        multihot_criteo.MultihotCriteo,
        multihot_criteo.pre_process_criteo_dlrm,
        multihot_criteo.DlrmPostProcess(),
        {"randomize": "total", "memory_map": True},
    ),
}

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "multihot-criteo",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    }
}


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="name of the mlperf model, ie. dlrm")
    parser.add_argument("--model-path", required=True, help="path to the model file")
    # parser.add_argument("--num-samples", type=int, required=True, help="number of samples to use for calibaration")
    # parser.add_argument("--upsample-rate", type=int, required=True, help="number of upsample rate to use for calibaration")
    # parser.add_argument("--num-bins", type=int, required=True, help="number of bins to use for calibaration")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument(
        "--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles"
    )
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument(
        "--max-batchsize", type=int, help="max batch size in a single inference"
    )
    parser.add_argument("--output", help="test results")
    parser.add_argument("--inputs", help="model inputs (currently not used)")
    parser.add_argument("--outputs", help="model outputs (currently not used)")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument(
        "--find-peak-performance",
        action="store_true",
        help="enable finding peak performance pass",
    )

    # file to use mlperf rules compliant parameters
    parser.add_argument(
        "--mlperf_conf", default="mlperf.conf", help="mlperf rules config"
    )
    # file for user LoadGen settings such as target QPS
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--duration", type=int, help="duration in milliseconds (ms)")
    parser.add_argument("--target-qps", type=int, help="target/expected qps")
    parser.add_argument(
        "--max-latency", type=float, help="mlperf max latency in pct tile"
    )
    parser.add_argument("--count-samples", type=int, help="dataset items to use")
    parser.add_argument("--count-queries", type=int, help="number of queries to use")
    parser.add_argument(
        "--samples-per-query-multistream",
        default=8,
        type=int,
        help="query length for multi-stream scenario (in terms of aggregated samples)",
    )
    # --samples-per-query-offline is equivalent to perf_sample_count
    parser.add_argument(
        "--samples-per-query-offline",
        type=int,
        default=2048,
        help="query length for offline scenario (in terms of aggregated samples)",
    )
    parser.add_argument(
        "--samples-to-aggregate-fix",
        type=int,
        help="number of samples to be treated as one",
    )
    parser.add_argument(
        "--samples-to-aggregate-min",
        type=int,
        help="min number of samples to be treated as one in random query size",
    )
    parser.add_argument(
        "--samples-to-aggregate-max",
        type=int,
        help="max number of samples to be treated as one in random query size",
    )
    parser.add_argument(
        "--samples-to-aggregate-quantile-file",
        type=str,
        help="distribution quantile used to generate number of samples to be treated as one in random query size",
    )
    parser.add_argument(
        "--samples-to-aggregate-trace-file",
        type=str,
        default="dlrm_trace_of_aggregated_samples.txt",
    )
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Whether calibration only for this run.",
    )
    parser.add_argument(
        "--int8-model-dir",
        type=str,
        default="./dlrm_int8.pt",
        help="int8 model location",
    )
    parser.add_argument("--use-int8", action="store_true", default=False)
    parser.add_argument("--use-bf16", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.numpy_rand_seed)

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")

    return args

features_in_hook = []
features_out_hook = []

def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)

def convert_int8_fx(
    max_batchsize: int,
    model: torch.nn.Module,
    int8_model_dir: str,
    ds,
):

    print("Quantizing the model using PT Quantizer")

    emb_qconfig = float_qparams_weight_only_qconfig_4bit
#    global_static_qconfig = QConfig(
#        activation=HistogramObserver.with_args(
#            qscheme=torch.per_tensor_affine,
#            dtype=torch.quint8,
#            upsample_rate=384,
#            bins=256,
#            quant_min=0,
#            quant_max=256,
#            reduce_range=False,
#        ),
#        weight=PerChannelMinMaxObserver.with_args(
#            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
#        ),
#    )
    global_static_qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            qscheme=torch.per_tensor_affine,
            dtype=torch.quint8,
            quant_min=0,
            quant_max=256,
            reduce_range=False,
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ),
    )
    static_qconfig_mapping = (
        QConfigMapping()
        .set_global(global_static_qconfig)
        .set_module_name(
            "sparse_arch.embedding_bag_collection.embedding_bags", emb_qconfig
        )
    )

    multi_hot = [
        3,
        2,
        1,
        2,
        6,
        1,
        1,
        1,
        1,
        7,
        3,
        8,
        1,
        6,
        9,
        5,
        1,
        1,
        1,
        12,
        100,
        27,
        10,
        3,
        1,
        1,
    ]
    dsx = torch.randn((max_batchsize, 13), dtype=torch.float)
    lsi = [torch.ones((max_batchsize * h), dtype=torch.long) for h in multi_hot]
    lso = [
        torch.arange(0, (max_batchsize + 1) * h, h, dtype=torch.long) for h in multi_hot
    ]

    model(dsx, lsi, lso)

    model = prepare_fx(model, static_qconfig_mapping, example_inputs=(dsx, lsi, lso))

    assert ds is not None
    count = ds.get_item_count()
    num_samples = 128000
    all_sample_ids = range(0, num_samples)
    ds.load_query_samples(all_sample_ids)
    # print("max_batch_size = ", max_batchsize)

    for i in range(0, num_samples, max_batchsize):
        sample_s = i
        sample_e = min(num_samples, i + max_batchsize)
        # print("sample_s = ", sample_s)
        # print("sample e = ", sample_e)
        densex, index, offset, labels = ds.test_data.load_batch(
            range(sample_s, sample_e)
        )
        model(densex, index, offset)

    model = convert_fx(model)
    model.eval()

    # ------------------------debug info ----------------------------
#    for name, module in model.named_modules():
#        if name == "sparse_arch.embedding_bag_collection.embedding_bags.16":
#            module.register_forward_hook(hook=hook)
#
#    model(dsx, lsi, lso)
#    print(f"features_in_hook[0] is {features_in_hook[0]}")
#    print(f"features_out_hook[0] shape is {features_out_hook[0].shape}")
#    print(f"features_out_hook[0] is {features_out_hook[0]}")
#    exit()
#
    model = torch.jit.trace(model, (dsx, lsi, lso), check_trace=True)
    model = torch.jit.freeze(model)
    model(dsx, lsi, lso)
    model(dsx, lsi, lso)
    torch.jit.save(model, int8_model_dir)
    print(f"calibration done and save to {int8_model_dir}")

    return model


def main():
    args = get_args()

    backend = get_backend(args.backend, args.dataset)
    wanted_dataset, pre_proc, _, kwargs = SUPPORTED_DATASETS[args.dataset]

    ds = wanted_dataset(
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
        data_path=args.dataset_path,
        name=args.dataset,
        pre_process=pre_proc,  # currently an identity function
        count=args.count_samples,
        samples_to_aggregate_fix=args.samples_to_aggregate_fix,
        samples_to_aggregate_min=args.samples_to_aggregate_min,
        samples_to_aggregate_max=args.samples_to_aggregate_max,
        samples_to_aggregate_quantile_file=args.samples_to_aggregate_quantile_file,
        samples_to_aggregate_trace_file=args.samples_to_aggregate_trace_file,
        max_ind_range=args.max_ind_range,
        **kwargs,
    )
    # load model to backend
    model = backend.load(args)
    # calibration
    if args.calibration:
        dlrm_model = model.model
        convert_int8_fx(
            args.max_batchsize,
            dlrm_model,
            args.int8_model_dir,
            ds,
        )

    # make one pass over the dataset to validate accuracy
    model = torch.jit.load(args.int8_model_dir)
    count = ds.get_item_count()
    # warmup
    results = np.zeros(count).astype(np.float32)
    targets = np.zeros(count).astype(np.float32)
    batchsize = args.max_batchsize

    for i in range(0, count, batchsize):
        sample_s = i
        sample_e = min(i + batchsize, count)
        densex, index, offset, labels = ds.val_data.load_batch(
            range(sample_s, sample_e)
        )
        r = model(densex, index, offset)
        results[sample_s:sample_e] = r.detach().cpu().numpy()
        targets[sample_s:sample_e] = labels.detach().cpu().float().numpy()
        print(f"Done for {i} / {count}", end="\r")

    print("\nTotal ROC AUC = ", sklearn.metrics.roc_auc_score(targets, results))


if __name__ == "__main__":
    main()
