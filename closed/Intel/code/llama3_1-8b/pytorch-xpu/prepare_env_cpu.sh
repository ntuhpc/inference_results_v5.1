pip install torch==2.6.0 torchaudio==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cpu && \
pip install pandas==2.2.2 toml==0.10.2 unidecode==1.3.8 inflect==7.3.1 librosa==0.10.2 py-libnuma==1.2 numpy==2.0.1 && \
pip install triton==3.1.0 && \
pip install setuptools-scm && \
pip install intel-extension-for-pytorch==2.6.0 && \
pip install auto-round==0.5.1
pip install llmcompressor==0.6.0.1

git clone https://github.com/vllm-project/vllm -b v0.8.5.post1
cd vllm
pip install -r requirements/cpu.txt
VLLM_TARGET_DEVICE=cpu python setup.py install
cd .. && rm -rf vllm
