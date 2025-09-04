cd /work #container's home dir

rm -rf closed/HPE/.dlrm-build-x86_64/ #This isn't cleaned up and causes issues in v5.0.2, manually remove before generating DLRM engines

#This script backs up build files before performing make clean
mv build/compliance_logs build/compliance_logs-$(date +%Y%m%d-%H%M%S)
mv build/full_results-backup build/full_results-backup-$(date +%Y%m%d-%H%M%S)
mv build/logs build/logs-$(date +%Y%m%d-%H%M%S)
mv build/submission-staging build/submission-staging-$(date +%Y%m%d-%H%M%S)
mv build/postprocessed_data build/postprocessed_data-$(date +%Y%m%d-%H%M%S)

#!!!CAUTION, this will delete previous performance logs!!!
make clean_shallow  # Make sure that the build/ directory isn't dirty
#!!!CAUTION, this will delete previous performance logs!!!

bash scripts/make_build.sh
