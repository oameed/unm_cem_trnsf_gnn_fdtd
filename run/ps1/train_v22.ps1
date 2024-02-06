
conda activate tf2pyWin

cd run\fdtd

echo ' MAKE DIRECTORY FOR EXPERIMENT v22'

$FileName=                            "..\..\networks\v22"
if (Test-Path $FileName) {
  rm -Recurse -Force $FileName
}
tar -xvzf ..\..\networks\v00.tar.gz -C ..\..\networks
mv        ..\..\networks\v00           ..\..\networks\v22

echo ' START TRAINING FOR EXPERIMENT v22'
python train_gnn.py -data type2 -net v22 -t 15 -eptr 2

echo ' GENERATING PREDICTIONS '
python predict.py   -data type2 -net v22 -t 15
matlab -batch "graphics('v22',3,'Power',0.5 ,0.05,0.5)" -noFigureWindows

cd ..\..\

conda deactivate tf2pyWin

