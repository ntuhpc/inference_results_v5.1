import code.common.constants as C
from importlib import import_module
dlrmv2_fields = import_module("code.dlrm-v2.tensorrt.fields")
import code.fields.harness as harness_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import copy



base = {
    # Data paths. You should not need to change this unless you know what you are doing.
    dlrmv2_fields.embeddings_path: '/home/mlperf_inf_dlrmv2/model/embedding_weights',
    model_fields.model_path: '/home/mlperf_inf_dlrmv2/model/model_weights',
    dlrmv2_fields.sample_partition_path: '/home/mlperf_inf_dlrmv2/criteo/day23/sample_partition.npy',
    harness_fields.tensor_path: '/home/mlperf_inf_dlrmv2/criteo/day23/fp16/day_23_dense.npy,/home/mlperf_inf_dlrmv2/criteo/day23/fp32/day_23_sparse_concatenated.npy',

    # Do not change
    harness_fields.use_graphs: False,
    model_fields.input_dtype: 'fp16',
    model_fields.input_format: 'linear',
    dlrmv2_fields.bot_mlp_precision: 'int8',
    dlrmv2_fields.embeddings_precision: 'int8',
    dlrmv2_fields.interaction_op_precision: 'int8',
    dlrmv2_fields.top_mlp_precision: 'int8',
    dlrmv2_fields.final_linear_precision: 'int8',
    model_fields.precision: 'int8',
    dlrmv2_fields.check_contiguity: True,
    harness_fields.coalesced_tensor: True,

    # Tune me!
    # If you are hitting GPU OOM, try reducing this value before reducing batch size
    dlrmv2_fields.embedding_weights_on_gpu_part: 1.0,
    
    # loadgen_fields.min_duration: 2400000,

    model_fields.gpu_batch_size: {
        'dlrm-v2': 600000,
    },
    loadgen_fields.offline_expected_qps: 500000,#120000,

    harness_fields.gpu_copy_streams: 1,
    harness_fields.gpu_inference_streams: 1,
    dlrmv2_fields.gpu_num_bundles: 2,

    # WARNING: Only enable this feature if you do satisfy the start_from_device MLCommons rules.
    harness_fields.start_from_device: True,
}

high_acc = copy.deepcopy(base)
high_acc[dlrmv2_fields.interaction_op_precision] = 'fp16'
high_acc[loadgen_fields.offline_expected_qps] = 350000 #120000

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base,
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): high_acc,
}