#!/bin/bash

# Run Benchmark (all scenarios)
SCENARIO=Offline MODE=Performance bash run_mlperf_llama3_1-8b.sh
SCENARIO=Server  MODE=Performance bash run_mlperf_llama3_1-8b.sh
SCENARIO=Offline MODE=Accuracy    bash run_mlperf_llama3_1-8b.sh
SCENARIO=Server  MODE=Accuracy    bash run_mlperf_llama3_1-8b.sh

# Run Compliance (all tests)
SCENARIO=Offline MODE=Compliance  bash run_mlperf_llama3_1-8b.sh
SCENARIO=Server  MODE=Compliance  bash run_mlperf_llama3_1-8b.sh

# Build submission
VENDOR=OEM SYSTEM=1-node-4x-BMG-Pro-B60-Dual bash scripts/prepare_submission.sh
