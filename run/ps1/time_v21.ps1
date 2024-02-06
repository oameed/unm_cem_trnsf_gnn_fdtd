
conda activate tf211pyWin

cd run\fdtd

echo ' MAKE DIRECTORY FOR EXPERIMENT v25'
$FileName=                            "..\..\networks\v25"
if (Test-Path $FileName) {
  rm -Recurse -Force $FileName
}
tar -xvzf ..\..\networks\v00.tar.gz -C ..\..\networks
mv        ..\..\networks\v00           ..\..\networks\v25

echo ' MAKE DIRECTORY FOR EXPERIMENT v26'
$FileName=                            "..\..\networks\v26"
if (Test-Path $FileName) {
  rm -Recurse -Force $FileName
}
tar -xvzf ..\..\networks\v00.tar.gz -C ..\..\networks
mv        ..\..\networks\v00           ..\..\networks\v26

echo ' MAKE DIRECTORY FOR EXPERIMENT v27'
$FileName=                            "..\..\networks\v27"
if (Test-Path $FileName) {
  rm -Recurse -Force $FileName
}
tar -xvzf ..\..\networks\v00.tar.gz -C ..\..\networks
mv        ..\..\networks\v00           ..\..\networks\v27


echo ' START TRAINING FOR EXPERIMENT v25'
python train_gnn.py -data type1 -net v25 -t 50 -eptr 2

echo ' START TRAINING FOR EXPERIMENT v26'
python train_gnn.py -data type1 -net v26 -t 25 -eptr 2

echo ' START TRAINING FOR EXPERIMENT v27'
python train_gnn.py -data type1 -net v27 -t 10 -eptr 2


echo ' GENERATING PREDICTIONS '
python predict.py   -data type1 -net v25 -t 50 -ex # 42.26 ms

echo ' GENERATING PREDICTIONS '
python predict.py   -data type1 -net v26 -t 25 -ex # 43.15 ms

echo ' GENERATING PREDICTIONS '
python predict.py   -data type1 -net v27 -t 10 -ex # 43.17 ms


cd ..\..\

conda deactivate tf211pyWin

