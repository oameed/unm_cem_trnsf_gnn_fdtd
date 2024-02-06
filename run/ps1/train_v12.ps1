
conda activate tf211pyWin

cd run\fdtd

echo ' MAKE DIRECTORY FOR EXPERIMENT v12'

$FileName=                            "..\..\networks\v12"
if (Test-Path $FileName) {
  rm -Recurse -Force $FileName
}
tar -xvzf ..\..\networks\v00.tar.gz -C ..\..\networks
mv        ..\..\networks\v00           ..\..\networks\v12

echo ' START TRAINING FOR EXPERIMENT v12'
python train_trnsf.py -data type2 -net v12 -t 30 -eptr 2

echo ' GENERATING PREDICTIONS '
python predict.py     -data type2 -net v12 -t 30 
matlab -batch "graphics('v12',3,'Power',0.5 ,0.05,1)" -noFigureWindows

cd ..\..\

conda deactivate tf211pyWin

