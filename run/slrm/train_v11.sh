#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --time=48:00:00
#SBATCH --partition=singleGPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=oameednkn@unm.edu
#SBATCH --job-name=TRNSF1

module load miniconda3
module load matlab

source  activate tf211py
cd      $SLURM_SUBMIT_DIR
cd      ../fdtd

echo   ' MAKE DIRECTORY FOR EXPERIMENT '                'v11'
rm     -rf                                ../../networks/v11
tar    -xzf  ../../networks/v00.tar.gz -C ../../networks
mv           ../../networks/v00           ../../networks/v11

echo   ' START TRAINING FOR EXPERIMENT '                'v11'
python train_trnsf.py -data type1 -net v11 -t 10 -sp 5 -eptr 500

echo ' GENERATING PREDICTIONS '
python predict.py     -data type1 -net v11 -t 10 -sp 5
matlab -nodisplay -nosplash -nodesktop -r "graphics('v11',3,'Power',0.5 ,0.05,1);exit;"

