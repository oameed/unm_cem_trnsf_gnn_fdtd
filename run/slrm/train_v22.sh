#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --time=48:00:00
#SBATCH --partition=singleGPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=oameednkn@unm.edu
#SBATCH --job-name=GNN2

module load miniconda3
module load matlab

source  activate tf2py
cd      $SLURM_SUBMIT_DIR
cd      ../fdtd

echo   ' MAKE DIRECTORY FOR EXPERIMENT '                'v22'
rm     -rf                                ../../networks/v22
tar    -xzf  ../../networks/v00.tar.gz -C ../../networks
mv           ../../networks/v00           ../../networks/v22

echo   ' START TRAINING FOR EXPERIMENT '                'v22'
python train_gnn.py -data type2 -net v22 -t 15 -eptr 250

cho ' GENERATING PREDICTIONS '
python predict.py   -data type2 -net v22 -t 15
matlab -nodisplay -nosplash -nodesktop -r "graphics('v22',3,'Power',0.5 ,0.05,0.5);exit;"


