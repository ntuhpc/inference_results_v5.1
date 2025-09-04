##These steps are from `./closed/NVIDIA/code/llama2-70b/tensorrt/README.md`
#source scripts/hpe_chippewa_falls.sh #get MLPERF_SCRATCH_PATH from hpe_<lab_location>.sh file
source scripts/hpe_grenoble.sh

cd ./closed/HPE/
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM && git checkout 0ab9d17a59c284d2de36889832fe9fc7c8697604 # Using 2/6/2024 ToT

make -C docker build
# The default docker command will not mount extra directory. If necessary, copy the docker command and append
# -v <src_dir>:<dst:dir> to mount your own directory.

#copy model since not sure how to mount model_dir using make docker run...
mkdir ./temp_model
mkdir ./temp_model/Llama2
cp -r $MLPERF_SCRATCH_PATH/models/Llama2/Llama-2-70b-chat-hf/ ./temp_model/Llama2

make -C docker run LOCAL_USER=1 #DOCKER_ARGS="-v $MLPERF_SCRATCH_PATH:/model" #--mount type=bind,source=$MLPERF_SCRATCH_PATH

##################################################
# The following steps should be performed within TRTLLM container. Change -a=90 to your target architecture
git lfs install && git lfs pull
python3 ./scripts/build_wheel.py -a=90 --clean --install --trt_root /usr/local/tensorrt/

# Quantize the benchmark, On L40s, you might need TP4
python ./examples/quantization/quantize.py --dtype=float16  --output_dir=./temp_model/Llama2/fp8-quantized-modelopt/llama2-70b-chat-hf-tp4pp1-fp8 --model_dir=./temp_model/Llama2/Llama-2-70b-chat-hf --qformat=fp8 --kv_cache_dtype=fp8 --tp_size 4

python ./examples/quantization/quantize.py --dtype=float16  --output_dir=./temp_model/Llama2/fp8-quantized-modelopt/llama2-70b-chat-hf-tp2pp1-fp8 --model_dir=./temp_model/Llama2/Llama-2-70b-chat-hf --qformat=fp8 --kv_cache_dtype=fp8 --tp_size 2

python ./examples/quantization/quantize.py --dtype=float16  --output_dir=./temp_model/Llama2/fp8-quantized-modelopt/llama2-70b-chat-hf-tp1pp1-fp8 --model_dir=./temp_model/Llama2/Llama-2-70b-chat-hf --qformat=fp8 --kv_cache_dtype=fp8 --tp_size 1


python ./examples/quantization/quantize.py --dtype=float16  --output_dir=./temp_model/Llama2/fp8-quantized-modelopt/llama2-70b-chat-hf-tp1pp2-fp8 --model_dir=./temp_model/Llama2/Llama-2-70b-chat-hf --qformat=fp8 --kv_cache_dtype=fp8 --tp_size 1 --pp_size 2

##################################################

#move temp_model back to scratch space
#cp -r ./temp_model/Llama2/fp8-quantized-ammo/ $MLPERF_SCRATCH_PATH/models/Llama2/.
cp -r ./temp_model/Llama2/fp8-quantized-modelopt $MLPERF_SCRATCH_PATH/models/Llama2/. #fp8-quantized-modelopt #v5.0 renamed fp8-quantized-ammo to fp8-quantized-modelopt
rm -rf ./temp_model

## Build and run the benchmarks
