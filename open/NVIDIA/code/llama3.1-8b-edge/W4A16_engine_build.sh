export ATTENTION_PLUGIN_PATH=/tmp/build/libAttentionPlugin.so
export INT4_GEMM_PLUGIN_PATH=/tmp/build/libInt4GemmPlugin.so

/tmp/build/examples/llm/llm_build  --onnxPath=/tmp/Llama-3.1-8B-INT4-ONNX/model.onnx --enginePath=/home/engines/8B_W4A16.engine --batchSize=1 --maxBatchSize=1 --maxInputLen=3072 --maxSeqLen=4096
