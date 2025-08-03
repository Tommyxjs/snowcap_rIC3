#!/bin/bash
#SBATCH --job-name=snowcap-rIC3
#SBATCH --ntasks=1
#SBATCH --exclusive         # 独占整台计算节点（或者指定独占资源）
#SBATCH --nodelist=node160
#SBATCH --cpus-per-task=16
#SBATCH --output=snowcap_rIC3_%j.log
#SBATCH --time=08:00:00
#SBATCH --mem=32G

# 切换到你的工作目录
cd /public/home/jwli/workSpace/xjs/sigcomm_exp_24/ver/snowcap_rIC3-main || {
  echo "Directory not found!" >&2
  exit 1
}

echo "My job ID is $SLURM_JOB_ID" | tee -a snowcap_rIC3.log
echo "Running on nodes:" | tee -a snowcap_rIC3.log
scontrol show hostnames $SLURM_JOB_NODELIST | tee -a snowcap_rIC3.log

echo "Begin time: $(date)" | tee -a snowcap_rIC3.log
NP=$SLURM_NTASKS

# 检查 run_experiment.py 是否存在
if [ ! -f run_experiment.py ]; then
  echo "Error: run_experiment.py not found!" | tee -a snowcap_rIC3.log
  exit 1
fi

echo "Running run_experiment.py" | tee -a snowcap_rIC3.log
python run_experiment.py | tee -a snowcap_rIC3.log
echo "End time: $(date)" | tee -a snowcap_rIC3.log
