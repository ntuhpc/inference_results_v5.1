Copyright 2025, MangoBoost, Inc. All rights reserved.

Use of the MangoBoost LLMBoost package for MLPerf is limited to MLPerf 
evaluation/benchmarking for any purposes (commercial or non-commercial) 
on clusters up to 8 nodes (64 GPUs in total) of MI300X, MI325X, MI350X, 
or MI355X AMD GPUs.

---

Please follow the instruction below to reproduce our result:

> Prerequisite: Make sure you have AMD base docker built beforehand.

### Build Final Docker
> Note: Please use the Dockerfile included in our MLPerf-v5.1 github repository. 

    cd <to-dockerfile-directory>
    docker compose build prod-rocm-mlperf-5_1-multi-node-mi35x-final

### Run the docker
    docker run -it --rm \
        --network host \
        --group-add video \
        --ipc host \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device=/dev/dri:/dev/dri \
        --device=/dev/kfd:/dev/kfd \
        -v <your quantized model>:/model/Llama-2-70b-chat-hf-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-MLPerf-GPTQ \
        -v <your data>:/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl \
        mb-llmboost-inference:prod-rocm-mlperf-5_1-multi-node-mi35x-final

### Reproduce Single Node Result

    ```
        cd /workspace/apps/mlperf
        export GPU_NAME=8x_mi355x
        bash run_single_node.sh
    ```

### Reproduce Multi-Node Result

    ```
        # Run Server on all the nodes
            cd /workspace/apps/mlperf
            vim cluster.json    # change to the client addr that will receive the response
            export GPU_NAME=64x_mi355x
            bash multi_node_server_offline.sh

        # Run Client on one of the nodes
            cd /workspace/apps/mlperf
            vim cluster.json    # change to the ip addr of the servers
            export GPU_NAME=64x_mi355x
            bash multi_node_client_offline.sh
    ```