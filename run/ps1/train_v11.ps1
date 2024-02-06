
conda activate tf211pyWin

cd run\fdtd

echo ' MAKE DIRECTORY FOR EXPERIMENT v11'

$FileName=                            "..\..\networks\v11"
if (Test-Path $FileName) {
  rm -Recurse -Force $FileName
}
tar -xvzf ..\..\networks\v00.tar.gz -C ..\..\networks
mv        ..\..\networks\v00           ..\..\networks\v11

echo ' START TRAINING FOR EXPERIMENT v11'
python train_trnsf.py -data type1 -net v11 -t 10 -sp 5 -eptr 500

echo ' GENERATING PREDICTIONS '
python predict.py     -data type1 -net v11 -t 10 -sp 5
matlab -batch "graphics('v11',3,'Power',0.5 ,0.05,1)" -noFigureWindows

cd ..\..\

conda deactivate tf211pyWin

