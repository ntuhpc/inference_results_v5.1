# ---------------------
# This script sets environment variables based on the GPU type, model, and scenario.
# ---------------------

# transport settings
export NCCL_SOCKET_IFNAME="<NCCL_SOCKET_IFNAME>"
export UCX_NET_DEVICES="<UCX_NET_DEVICES>"
export NCCL_IB_HCA="<NCCL_IB_HCA>"

# ---------- model-specific parameters ----------
if [ "$MODEL" = "llama2-70b" ]; then
    export PERFORMANCE_SAMPLE_COUNT="24576"
    export TENSOR_PATH="/data/llama2-70b/"
    export LLM_GEN_CONFIG_PATH="code/llama2-70b/tensorrt/generation_config.json"
    export DATASET_CLS="code.llama2-70b.tensorrt.dataset.LlamaDataset"

    # accuracy testing
    export HUGGINGFACE_CHECKPOINT="/work/build/models/Llama2/Llama-2-70b-chat-hf/"
    export DATASET_PATH="/work/build/data/llama2-70b/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"

elif [ "$MODEL" = "llama3_1-405b" ]; then
    export PERFORMANCE_SAMPLE_COUNT="8313"
    export TENSOR_PATH="/data/preprocessed_data/llama3.1-405b/"
    export LLM_GEN_CONFIG_PATH="code/llama3_1-405b/tensorrt/generation_config.json"
    export DATASET_CLS="code.llama3_1-405b.tensorrt.dataset.Llama3_1Dataset"

    # accuracy testing
    export HUGGINGFACE_CHECKPOINT="/work/build/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct/"
    export DATASET_PATH="/work/build/data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl"
fi

# ---------- GPU-type parameters ----------
if [ "$GPU_TYPE" = "H100" ]; then
    export RANKFILE="scripts/NUMA/DGX-H100_H100-HBM3-80GB.hostfile"
    export CONT_NAME="<container_name>"
else
    export RANKFILE="scripts/NUMA/DGX-H200_H200-PCIe-141GB.hostfile"
    export CONT_BASE_NAME="<container_name>"
fi

# ---------- combined parameters ----------
if [ "$GPU_TYPE" = "H100" ]; then
    if [ "$MODEL" = "llama2-70b" ]; then
        # Server for llama2-70b
        if [ "$SCENARIO" = "Server" ]; then
            export ENGINE_PATH="./build/engines/DGX_H100_H100-SXM-80GBx8/llama2-70b/Server/llama2-70b-Server-gpu-b2048-fp8-tp2pp1-custom_k_99_9_MaxP/rank0.engine"
            export BATCH_SIZE=2048
            export MLPERF_CONF_PATH="<mlperf.conf>"
            export USER_CONF_PATH="<user.conf>"
            export LOG_OUTPUT="${LOG_DIR_BASE}/DGX-H100_H100-HBM3-80GBx${NUM_GPUS}_TRT/llama2-70b-99.9/Server/${TEST_MODE}/"
            export TRTLLM_BUILD_FLAGS="max_beam_width:1,kv_cache_type:paged,remove_input_padding:enable,multiple_profiles:enable,use_fused_mlp:enable,context_fmha:enable,max_num_tokens:1024,max_input_len:1024,max_seq_len:2048,use_fp8_context_fmha:enable,use_paged_context_fmha:enable,tokens_per_block:32,tensor_parallelism:2,pipeline_parallelism:1,reduce_fusion:enable,gemm_swiglu_plugin:fp8"
            export TRTLLM_RUNTIME_FLAGS="enable_block_reuse:true,exclude_input_from_output:true,use_inflight_batching:true,max_num_tokens:1024,batch_scheduler_policy:max_util,context_chunking_policy:first_come_first_served,kvcache_free_gpu_mem_frac:0.95,enable_chunked_context:true"
        fi
        # Offline for llama2-70b
        if [ "$SCENARIO" = "Offline" ]; then
            export ENGINE_PATH="./build/engines/DGX-H100_H100-SXM-80GBx8/llama2-70b/Offline/llama2-70b-Offline-gpu-b1024-fp8-tp1pp2-custom_k_99_9_MaxP/rank0.engine" 
            export BATCH_SIZE=1024
            export MLPERF_CONF_PATH="<mlperf.conf>"
            export USER_CONF_PATH=="<user.conf>"
            export LOG_OUTPUT="${LOG_DIR_BASE}/DGX-H100_H100-HBM3-80GBx${NUM_GPUS}_TRT/llama2-70b-99.9/Offline/${TEST_MODE}/"
            export TRTLLM_BUILD_FLAGS="max_beam_width:1,kv_cache_type:paged,remove_input_padding:enable,multiple_profiles:enable,use_fused_mlp:enable,context_fmha:enable,max_num_tokens:1024,max_input_len:1024,max_seq_len:2048,use_fp8_context_fmha:enable,use_paged_context_fmha:enable,tokens_per_block:32,tensor_parallelism:1,pipeline_parallelism:2,reduce_fusion:enable,gemm_swiglu_plugin:fp8"
            export TRTLLM_RUNTIME_FLAGS="enable_block_reuse:true,exclude_input_from_output:true,use_inflight_batching:true,max_num_tokens:1024,batch_scheduler_policy:max_util,context_chunking_policy:first_come_first_served,kvcache_free_gpu_mem_frac:0.95,enable_chunked_context:true"
        fi
    elif [ "$MODEL" = "llama3_1-405b" ]; then
        # Server for llama3_1-405b
        if [ "$SCENARIO" = "Server" ]; then
            export ENGINE_PATH="./build/engines/DGX-H100_H100-SXM-80GBx8/llama3_1-405b/Server/llama3_1-405b-Server-gpu-b512-fp8-tp8pp1-custom_k_99_MaxP/rank0.engine"
            export BATCH_SIZE=256
            export MLPERF_CONF_PATH="<mlperf.conf>"
            export USER_CONF_PATH="<user.conf>"
            export LOG_OUTPUT="${LOG_DIR_BASE}/DGX-H100_H100-HBM3-80GBx${NUM_GPUS}_TRT/llama3_1-405b-70b-99.9/Server/${TEST_MODE}/"
            export TRTLLM_BUILD_FLAGS="max_beam_width:1,kv_cache_type:paged,remove_input_padding:enable,multiple_profiles:enable,use_fused_mlp:enable,context_fmha:enable,max_num_tokens:8192,max_input_len:20000,max_seq_len:22000,use_paged_context_fmha:enable,tokens_per_block:32,use_fp8_context_fmha:enable,tensor_parallelism:8,pipeline_parallelism:1,gemm_allreduce_plugin:float16"
            export TRTLLM_RUNTIME_FLAGS="exclude_input_from_output:true,use_inflight_batching:true,max_num_tokens:2560,batch_scheduler_policy:max_util,context_chunking_policy:first_come_first_served,kvcache_free_gpu_mem_frac:0.9,enable_chunked_context:true,max_batch_size:64"
        fi
        # Offline for llama3_1-405b
        if [ "$SCENARIO" = "Offline" ]; then
            export ENGINE_PATH="./build/engines/DGX-H100_H100-SXM-80GBx8/llama3_1-405b/Offline/llama3_1-405b-Offline-gpu-b256-fp8-tp8pp1-custom_k_99_MaxP/rank0.engine"
            export BATCH_SIZE=256
            export MLPERF_CONF_PATH="<mlperf.conf>"
            export USER_CONF_PATH="<user.conf>"
            export LOG_OUTPUT="${LOG_DIR_BASE}/DGX-H100_H100-HBM3-80GBx${NUM_GPUS}_TRT/llama3_1-405b-70b-99.9/Offline/${TEST_MODE}/"
            export TRTLLM_BUILD_FLAGS="max_beam_width:1,kv_cache_type:paged,remove_input_padding:enable,multiple_profiles:enable,use_fused_mlp:enable,context_fmha:enable,max_num_tokens:2560,max_input_len:20000,max_seq_len:22000,use_paged_context_fmha:enable,tokens_per_block:32,use_fp8_context_fmha:enable,tensor_parallelism:8,pipeline_parallelism:1"
            export TRTLLM_RUNTIME_FLAGS="exclude_input_from_output:true,use_inflight_batching:true,max_num_tokens:1536,batch_scheduler_policy:max_util,context_chunking_policy:first_come_first_served,kvcache_free_gpu_mem_frac:0.95,enable_chunked_context:true"
        fi
    fi
# H200 variables
else
    if [ "$MODEL" = "llama2-70b" ]; then
        # Server for llama2-70b
        if [ "$SCENARIO" = "Server" ]; then
            export ENGINE_PATH="./build/engines/C885A_M8_H200_SXM_141GBx8/llama2-70b/Server/llama2-70b-Server-gpu-b2048-fp8-tp1pp1-custom_k_99_9_MaxP/rank0.engine"
            export BATCH_SIZE=2048
            export MLPERF_CONF_PATH="<mlperf.conf>"
            export USER_CONF_PATH="<user.conf>"
            export LOG_OUTPUT="${LOG_DIR_BASE}/C885A_M8_H200_SXM_141GBx${NUM_GPUS}_TRT/llama2-70b-99.9/Server/${TEST_MODE}/"
            export TRTLLM_BUILD_FLAGS="max_beam_width:1,kv_cache_type:paged,remove_input_padding:enable,multiple_profiles:enable,use_fused_mlp:enable,context_fmha:enable,max_num_tokens:1536,max_input_len:1024,max_seq_len:2048,use_fp8_context_fmha:enable,use_paged_context_fmha:enable,tokens_per_block:32,tensor_parallelism:1,pipeline_parallelism:1,gemm_swiglu_plugin:fp8"
            export TRTLLM_RUNTIME_FLAGS="exclude_input_from_output:true,use_inflight_batching:true,max_num_tokens:1536,batch_scheduler_policy:max_util,context_chunking_policy:first_come_first_served,kvcache_free_gpu_mem_frac:0.9,enable_chunked_context:true"
        fi
        # Offline for llama2-70b
        if [ "$SCENARIO" = "Offline" ]; then
            export ENGINE_PATH="./build/engines/C885A_M8_H200_SXM_141GBx8/llama2-70b/Offline/llama2-70b-Offline-gpu-b2048-fp8-tp1pp1-custom_k_99_9_MaxP/rank0.engine" 
            export BATCH_SIZE=2048
            export MLPERF_CONF_PATH="<mlperf.conf>"
            export USER_CONF_PATH="<user.conf>"
            export LOG_OUTPUT="${LOG_DIR_BASE}/C885A_M8_H200_SXM_141GBx${NUM_GPUS}_TRT/llama2-70b-99.9/Offline/${TEST_MODE}/"
            export TRTLLM_BUILD_FLAGS="max_beam_width:1,kv_cache_type:paged,remove_input_padding:enable,multiple_profiles:enable,use_fused_mlp:enable,context_fmha:enable,max_num_tokens:1536,max_input_len:1024,max_seq_len:2048,use_fp8_context_fmha:enable,use_paged_context_fmha:enable,tokens_per_block:32,tensor_parallelism:1,pipeline_parallelism:1,gemm_swiglu_plugin:fp8"
            export TRTLLM_RUNTIME_FLAGS="exclude_input_from_output:true,use_inflight_batching:true,max_num_tokens:1536,batch_scheduler_policy:max_util,context_chunking_policy:first_come_first_served,kvcache_free_gpu_mem_frac:0.9,enable_chunked_context:true"

        fi
    elif [ "$MODEL" = "llama3_1-405b" ]; then
        # Server for llama3_1-405b
        if [ "$SCENARIO" = "Server" ]; then
            export ENGINE_PATH="./build/engines/C885A_M8_H200_SXM_141GBx8/llama3_1-405b/Server/llama3_1-405b-Server-gpu-b512-fp8-tp8pp1-custom_k_99_MaxP/rank0.engine"
            export BATCH_SIZE=512
            export MLPERF_CONF_PATH="<mlperf.conf>"
            export USER_CONF_PATH="<user.conf>"
            export LOG_OUTPUT="${LOG_DIR_BASE}/C885A_M8_H200_SXM_141GBx${NUM_GPUS}_TRT/llama3_1-405b-99.9/Server/${TEST_MODE}/"
            export TRTLLM_BUILD_FLAGS="max_beam_width:1,kv_cache_type:paged,remove_input_padding:enable,multiple_profiles:enable,use_fused_mlp:enable,context_fmha:enable,max_num_tokens:8192,max_input_len:20000,max_seq_len:22000,use_paged_context_fmha:enable,tokens_per_block:32,use_fp8_context_fmha:enable,tensor_parallelism:8,pipeline_parallelism:1,gemm_allreduce_plugin:float16"
            export TRTLLM_RUNTIME_FLAGS="exclude_input_from_output:true,use_inflight_batching:true,max_num_tokens:2560,batch_scheduler_policy:max_util,context_chunking_policy:first_come_first_served,kvcache_free_gpu_mem_frac:0.9,enable_chunked_context:true,max_batch_size:64"
        fi
        # Offline for llama3_1-405b
        if [ "$SCENARIO" = "Offline" ]; then
            export ENGINE_PATH="./build/engines/C885A_M8_H200_SXM_141GBx8/llama3_1-405b/Offline/llama3_1-405b-Offline-gpu-b512-fp8-tp4pp2-custom_k_99_MaxP/rank0.engine"
            export BATCH_SIZE=512
            export MLPERF_CONF_PATH="<mlperf.conf>"
            export USER_CONF_PATH="<user.conf>"
            export LOG_OUTPUT="${LOG_DIR_BASE}/C885A_M8_H200_SXM_141GBx${NUM_GPUS}_TRT/llama3_1-405b-99.9/Offline/${TEST_MODE}/"
            export TRTLLM_BUILD_FLAGS="max_beam_width:1,kv_cache_type:paged,remove_input_padding:enable,multiple_profiles:enable,use_fused_mlp:enable,context_fmha:enable,max_num_tokens:2560,max_input_len:20000,max_seq_len:22000,use_paged_context_fmha:enable,tokens_per_block:32,use_fp8_context_fmha:enable,tensor_parallelism:4,pipeline_parallelism:2,gemm_allreduce_plugin:float16"
            export TRTLLM_RUNTIME_FLAGS="exclude_input_from_output:true,use_inflight_batching:true,max_num_tokens:1536,batch_scheduler_policy:max_util,context_chunking_policy:first_come_first_served,kvcache_free_gpu_mem_frac:0.95,enable_chunked_context:true"
        fi
    fi
fi