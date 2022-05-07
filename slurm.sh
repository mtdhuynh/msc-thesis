#!/bin/bash
#
#SBATCH -p gpulong
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:20gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#
#SBATCH -J training
#SBATCH -o /home/thuynh/ms-thesis/logs/%j_%x.txt
module load cuda-11.4
module load anaconda

echo "[SLURM] Job $SLURM_JOB_ID start time: $(date +"%T")"
echo "[SLURM] Allocated GPUs: $SLURM_GPUS_ON_NODE"

source activate ms-thesis
cd /home/thuynh/ms-thesis
python3 src/train.py --device cuda --config data/04_model_input/config/retinanet.yaml --no-amp

echo "[SLURM] Job $SLURM_JOB_ID end time: $(date +"%T")"