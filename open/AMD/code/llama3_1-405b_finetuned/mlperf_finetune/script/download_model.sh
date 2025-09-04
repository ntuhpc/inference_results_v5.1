pip install huggingface

export SSL_CERT_DIR='/etc/ssl/certs'
export REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'

huggingface-cli download \
    --resume-download meta-llama/Llama-3.1-405B-Instruct \
    --local-dir /model/Llama-3.1-405B-Instruct \
    --local-dir-use-symlinks False \
    --exclude *consolidated*