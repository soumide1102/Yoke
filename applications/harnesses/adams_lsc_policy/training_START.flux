#!/bin/bash

# This is a setup for GPU training on RZadams.

# The following are one set of Flux options for the RZadams partitions.

# RZadams has 96 CPUs/node and 4 GPUs/node

#flux: --job-name=ksh_test
#flux: --time=20m
#flux: --queue=pdebug
#flux: --nodes=1
#flux: --ntasks-per-node=4
#flux: --gpus-per-task=1
#flux: --cores-per-task=24
#flux: --output=ksh_test.out
#flux: --error=ksh_test.err
#flux: --verbose


# # Set the master node's address and port
# MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# MASTER_PORT=$(shuf -i 1024-65535 -n 1)  # Choose a random port
# export MASTER_ADDR MASTER_PORT

# # Check available GPUs
# sinfo  -o "%P %.24G %N"
# srun /usr/bin/echo $CUDA_AVAILABLE_DEVICES
# nvidia-smi

# # Specify NCCL communication
# export NCCL_SOCKET_IFNAME=hsn0  # Check possible interfaces with `ip link show`

# Debugging distributed data parallel
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT

# Load correct conda environment
export ROCMVERSION=6.2.4
module load rocm/$ROCMVERSION
module load cray-python/3.11.7
source yoke_250415/bin/activate

# # Set number of threads per GPU
# export OMP_NUM_THREADS=10

# # MKL problems
# export MKL_SERVICE_FORCE_INTEL=1

# Get start time
export date00=`date`

# Start the Code
# Explicitly set TCP environment for the following...
flux run python -m "import torch; x = torch.rand(3, 3); print(x)"
flux run python -m "import torch; print(torch.cuda.is_available())"

#srun python -u <train_script> @study<studyIDX>_START.input

# Get end time and print to stdout
export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01