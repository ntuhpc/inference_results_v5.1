# The current build only supports SM89/SM90. If you want to try SM80 support, please go to Makefile.build and modify the "-a=90" flag from "build_trt_llm" target.
cd /work
make build_trt_llm
BUILD_TRTLLM=1 make build_harness

##then do make generate_engines ... commands


