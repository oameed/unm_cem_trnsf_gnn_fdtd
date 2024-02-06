#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --time=48:00:00
#SBATCH --partition=singleGPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=oameednkn@unm.edu
#SBATCH --job-name=GNN3

module load miniconda3
module load matlab

source  activate tf2py
cd      $SLURM_SUBMIT_DIR
cd      ../fdtd

echo   ' MAKE DIRECTORY FOR EXPERIMENT '                'v23'
rm     -rf                                ../../networks/v23
tar    -xzf  ../../networks/v00.tar.gz -C ../../networks
mv           ../../networks/v00           ../../networks/v23

echo   ' START TRAINING FOR EXPERIMENT '                'v23'
python train_gnn.py -data type3 -net v23 -t 15 -eptr 250

cho ' GENERATING PREDICTIONS '
python predict.py   -data type3 -net v23 -t 15
matlab -nodisplay -nosplash -nodesktop -r "graphics('v23',3,'Power',0.5 ,0.05,0.5);exit;"

