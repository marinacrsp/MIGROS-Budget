#!/bin/bash
#SBATCH --output=/work/scratch/mcrespo/output/%j.out
#SBATCH --account=dl
#SBATCH --partition=gpu
#SBATCH --nodes=1

# Load Conda

module add cuda/12.1

source /home/mcrespo/miniconda3/etc/profile.d/conda.sh
conda activate /home/mcrespo/miniconda3/envs/sel_py11
nvcc --version

# python selora_training.py
python classifier_train.py --main=/home/mcrespo/migros_deepL/BraTS2021_final/ --test=sample_flair_test150/ --train=sample_flair_train160/ --syn=sample_output_selora1.08/ --merge=False --epochs=50 --lr=0.0001 --seed 104
# python compute_fid.py
