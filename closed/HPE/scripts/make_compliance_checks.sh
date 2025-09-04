########## Run compliance on all models #############
cd /work #container's home dir
SUBMITTER="HPE"
export SUBMITTER=$SUBMITTER

#Setup for audit
make stage_results SUBMITTER=$SUBMITTER 

##Run audit checks
#make run_audit_harness SUBMITTER=$SUBMITTER 
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=resnet50 --scenarios=offline,server"
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=retinanet --scenarios=offline,server" 
#make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=rnnt --scenarios=offline,server"
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=3d-unet --scenarios=offline --config_ver=default,high_accuracy" 
#make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=bert --scenarios=offline,server --config_ver=default,high_accuracy"
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=dlrm-v2 --scenarios=offline,server --config_ver=default,high_accuracy"
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=gptj --scenarios=offline,server --config_ver=default,high_accuracy"
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=stable-diffusion-xl --scenarios=offline,server"
#make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=llama2-70b --scenarios=offline,server --config_ver=default,high_accuracy"
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=llama2-70b --scenarios=offline,server,interactive --config_ver=default,high_accuracy"
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=offline,server"
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=llama3.1-8b --scenarios=offline,server,interactive"
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=mixtral-8x7b --scenarios=offline,server"
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=whisper --scenarios=offline"
make run_audit_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=rgat --scenarios=offline,server"

#print results
echo "Audit tests finished!"
echo "Benchmarks failing compliance:"
grep -r "TEST FAIL" build/compliance_logs/
echo "Benchmarks passing compliance:"
grep -r "TEST PASS" build/compliance_logs/

#once all above pass, run this:
make stage_compliance SUBMITTER=$SUBMITTER
