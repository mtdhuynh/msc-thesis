#!/bin/bash
#
#SBATCH -p small
#
#SBATCH -J training
#SBATCH -o /home/thuynh/ms-thesis/logs/%j_%x.txt
#SBATCH -t 99:00:00

echo "SLURM Job $SLURM_JOB_ID start time: $(date +"%T")"

cd /home/thuynh/ms-thesis
python3 src/main.py

echo "SLURM Job $SLURM_JOB_ID end time: $(date +"%T")"