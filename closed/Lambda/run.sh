# Check if benchmark name is passed
if [ -z "$1" ]; then
  read -p "Enter benchmark name : " BENCHMARK
else
  BENCHMARK="$1"
fi


make run_harness RUN_ARGS="--benchmarks=${BENCHMARK} --scenarios=Offline --test_mode=AccuracyOnly" && \
make run_harness RUN_ARGS="--benchmarks=${BENCHMARK} --scenarios=Server --test_mode=AccuracyOnly" && \
make run_harness RUN_ARGS="--benchmarks=${BENCHMARK} --scenarios=Offline --test_mode=PerformanceOnly" && \
make run_harness RUN_ARGS="--benchmarks=${BENCHMARK} --scenarios=Server --test_mode=PerformanceOnly"

export LOG_DIR=/work/build/logs

echo "_________________________________________________________________________________"
echo " *********       STAGE RESULTS                                       ************"
echo "_________________________________________________________________________________"
BENCHMARKS=${BENCHMARK} make stage_results

echo "*********************************************************************************"
echo " *********              RUN AUDIT                                    ************"
echo "*********************************************************************************"

echo make run_audit_harness RUN_ARGS="--benchmarks=${BENCHMARK} --scenarios=Offline"
echo make run_audit_harness RUN_ARGS="--benchmarks=${BENCHMARK} --scenarios=Server"
