#!/bin/bash -l
#SBATCH --nodes=1
# #SBATCH --array=0-19
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH -G h100:1
#SBATCH --mem=400gb
#SBATCH --time=7-00:15:00 
#SBATCH --job-name=rg-evo-run
#SBATCH --mail-user=jlang15@uni-koeln.de
#SBATCH --mail-type=ALL
#SBATCH --output=logfile.out
#SBATCH --error=error.out

# Provides necessary environment for GPU 
module purge
module load accel/nvhpc/24.9

# When using OpenMP Offloading. It makes sure that kernels are running on the GPU
export OMP_TARGET_OFFLOAD=MANDATORY

# Gives information about data mapping and kernel launch
export NVCOMPILER_ACC_NOTIFY=3

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

presendir=$PWD
echo $presendir

# To organise the work directory according to date, time, and job ID 
dt=`date '+%Y-%m-%d_%H%M'`
workdir=/scratch/${USER}/GPU_${SLURM_JOB_ID}_$dt
echo "Working directory: $workdir"

mkdir -p $workdir

rsync -a ./ "$workdir/"

cd $workdir

# nsys profile --trace=openmp,openacc,cuda -o profile.out --stats=true ./StreamCluster
# ncu --export my_report --kernel-name indexMatAllKernel_specialized --launch-skip 0 --launch-count 1 "./StreamCluster"
# ncu --set full -o profile StreamCluster
# ncu --set full --indexMatAllKernel_specialized all --export my_report ./StreamCluster
USE_PROFILING=false
USE_NCU=false

# l=$(awk "BEGIN { printf \"%.2f\", 0.05 * $SLURM_ARRAY_TASK_ID }")
# echo "[$(date)] Starting RG-Evo with -l $l"

rm -rf build && ./build.sh --clean
  
if $USE_PROFILING; then
  nsys profile --trace=openmp,openacc,cuda -o profile.out ./RG-Evo
elif $USE_NCU; then
  ncu --set full -o profile ./RG-Evo
else
  # ./RG-Evo -m 10000 -D false -l "$l"
  ./RG-Evo -p 2 -q 6 -l 0.9 -L 512 -m 1e8 -t 1e7 -D false -S false
fi

# Move error and logfile to work directory
mv $presendir/*.out $workdir

# To keep the script in case it is modified for the next job
cp $presendir/job.sh $workdir

echo "Job finished at $(date)"
echo "Output stored in $workdir"
