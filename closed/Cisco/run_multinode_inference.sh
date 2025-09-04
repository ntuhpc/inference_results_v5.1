#-------------------------------------------

# 8, 16, or 32
export NUM_GPUS=16

# Server or Offline
export SCENARIO="Server"

# AccuracyOnly or PerformanceOnly
export TEST_MODE="PerformanceOnly"

# llama3_1-405b or llama2-70b
export MODEL="llama2-70b"

# H100 or H200
export GPU_TYPE="H100"

#-------------------------------------------

bash start_test.sh
