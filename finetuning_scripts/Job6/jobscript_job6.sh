#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=a100:2
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --account=swabhas_1457

module purge
module load gcc git nvhpc python/3.12.2

python -m venv repairllamaenv1
source ./repairllamaenv1/bin/activate
python -m pip install -r requirements.txt

python finetune_job6.py