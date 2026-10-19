#!/bin/bash
#SBATCH --job-name=C2_Crack_TO2_Delta
#SBATCH --account=bhzz-delta-gpu         # Delta GPU account (50 h balance)
#SBATCH --partition=gpuA100x4            # A100 40GB, 4 GPUs/node, shared (48h max walltime)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1                # one A100
#SBATCH --cpus-per-task=16               # 1/4 of the 64-core node (proportional to 1 GPU)
#SBATCH --mem=64g                        # 1/4 of the 256 GB node
#SBATCH --time=8:00:00                  # 8-hour wall clock (C2 250K)
#SBATCH --exclude=gpua021                # skip the node that NODE_FAIL'd
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
### uncomment for email:
### #SBATCH --mail-user=you@illinois.edu
### #SBATCH --mail-type=END,FAIL

set -e

# --- send all job output straight into this case's Results folder. Slurm's
#     --output cannot create directories, so (like the Sophia PBS) we mkdir
#     the case folder and exec-redirect stdout/stderr into it here. ---
CASE_DIR="/u/xsun7/Aug_21_2D/Results/Crack/C2_Crack_TO2_Delta"
mkdir -p "${CASE_DIR}"
exec > "${CASE_DIR}/C2_Crack_TO2_Delta.out" 2> "${CASE_DIR}/C2_Crack_TO2_Delta.err"
module reset                             # start from Delta's clean default modules

source ~/envs/TO/bin/activate            # your venv (torch bundles its own CUDA)
cd /u/xsun7/Aug_21_2D                     # base_folder (matches the .py)

# cap CPU BLAS/OpenMP threads to the allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

echo "node=$(hostname)  job=$SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

# Single GPU -> plain python (driver's world_size=1 path; Slurm exposes only the
# 1 allocated A100, so get_multiGPU(1) sees exactly 1 GPU).
python C2_Crack_TO2_Delta.py
