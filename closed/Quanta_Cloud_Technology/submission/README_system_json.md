## Summary of mandatory JSON Fields for submission system.json

This file contains a guide for filling the system json file. It's advised to fill out the [dummy_system.json](./submission/dummy_system.json) file prior to running the submission and it will be copied to the proper places.

**accelerator_host_interconnect**:
   - **Source**: [AMD Instinct MI300X spec](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)/[AMD Instinct MI325X spec](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
   - **Value**: `PCIe Gen5 x16` for both

**accelerator_interconnect**:
   - **Command**: `sudo dmesg | grep -i xgmi`
   - **Description**: This command checks the system message buffer for entries related to XGMI (Infinity Fabric), which is the interconnect technology used in AMD accelerators.
   - **Value**: `XGMI`

**accelerator_memory_capacity**:
   - **Source**: [AMD Instinct MI300X spec](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)/[AMD Instinct MI325X spec](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
   - **Value**: `192GB`/`256GB`

**accelerator_memory_configuration**:
   - **Source**: [AMD Instinct MI300X spec](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)/[AMD Instinct MI325X spec](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
   - **Value**: `HBM3`/`HBM3E`

**accelerator_model_name**:
   - **Source**: [AMD Instinct MI300X spec](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)/[AMD Instinct MI325X spec](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
   - **Value**: `AMD Instinct MI325X`/`AMD Instinct MI300X`

**accelerators_per_node**:
   - **Value**: `8`

**cooling**:
   - **Source**: [AMD Instinct MI300X spec](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)/[AMD Instinct MI325X spec](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
   - **Value**: `HBM3`/`HBM3E`

**division**:
   - **Value**: `closed`

**framework**:
   - **Commands**: `pip list | grep vllm`, `pip list | grep torch`, `apt show rocm-libs | grep 'Version'`
   - **Values**: `vLLM 0.0.0.dev1+mlperf50, Pytorch 2.7.0a0+git3a58512, ROCm 6.3.1`

**host_memory_capacity**:
   - **Command**: `sudo lshw -C memory`
   - **Description**: This command lists detailed information about the host memory configuration, including capacity and type. Check the `*-memory:0` section, here you will find the capacity

**host_memory_configuration**:
   - **Commands**: `sudo lshw -C memory | grep 'bank' | wc -l`, `sudo lshw -C memory | grep description | uniq`
   - **Description**: First command counts the number of accelerators second command gets the name of the memory
   - **Example**: `24x 96GiB Samsung M321RYGA0PB0-CWMXJ`

**host_network_card_count**:
   - **Commands**: `sudo lshw -class network | grep capacity`
   - **Description**: Use the above command to identify the bandwidth of network cards for each type.
   - **Example**: `2x 1Gbit, 4x 25Gbit`

**host_networking**:
   - **Commands**: `sudo lshw -class network`
   - **Description**: Use the above command to identify the product name of network cards and their bandwidth.
   - **Example**: `2x NetXtreme BCM5720 Gigabit Ethernet PCIe, 4x BCM57504 NetXtreme-E 10Gb/25Gb/40Gb/50Gb/100Gb/200Gb Ethernet`

**host_networking_topology**:
   - **Commands**: `sudo lshw -class network`
   - **Description**: Use the above command to identify the type of networking topology for each card type.
   - **Examples**: `Ethernet on switching network; Ethernet/Infiniband on switching network; Infiniband on peer to peer network; USB forwarded`

**host_processor_core_count**:
   - **Command**: `lscpu | grep 'Core(s) per socket'`
   - **Description**: This command retrieves the number of cores per socket for the host processor.

**host_processor_model_name**:
 - **Command**: `lscpu | grep 'Model name'`
 - **Description**: This command fetches the model name of the host processor.

**host_processors_per_node**:
   - **Command**: `lscpu | grep 'Socket(s)'`
   - **Description**: This command checks how many processor sockets are available per node, indicating the number of processors installed.

**host_storage_capacity**:
   - **Command**: `sudo fdisk -l`, `lsblk -o NAME,TYPE,SIZE,ROTA,MODEL`
   - **Description**: The first command lists all disk partitions and their sizes, while the second command provides a detailed view of block devices, including their names, types, sizes, and models.

**host_storage_type**:
   - **Command**: `sudo fdisk -l`, `lsblk -o NAME,TYPE,SIZE,ROTA,MODEL`
   - **Description**: Similar to the previous commands, these commands help identify the type of storage used
   - **Example**: `NVMe SSD`

**number_of_nodes**:
   - **Value**: `1`

**operating_system**:
   - **Command**: `lsb_release -a`
   - **Description**: This command displays detailed information about the operating system, including its version and codename.

**other_software_stack**:
   - **Command**: `pip list | grep vllm`, `dpkg -l | grep hipblaslt`, `pip list | grep flash_attn`
   - **Description**: The first command lists the version of vLLM. The second command lists installed packages related to hipBLASLT, while the third checks for the presence of the Flash Attention library in the Python environment.
   - **Example**: `vllm 0.0.0.dev1+mlperf50, hipblaslt 0.13.0-6b6a7243, flash_attn 2.7.2`

**status**:
   - **Value**: `available`

**submitter**:
   - **Description**: Name of the submitter company
   - **Example**: `AMD`

**system_name**:
   - **Command**: `sudo lshw -short | grep system`
   - **Description**: This command retrieves a short summary of the system hardware, including the system name.
