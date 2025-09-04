#!/bin/bash
set -e

script_dir=$(dirname -- $0)

bash ${script_dir}/quantize_mixtral_8x7b_offline_model.sh
bash ${script_dir}/quantize_mixtral_8x7b_server_model.sh