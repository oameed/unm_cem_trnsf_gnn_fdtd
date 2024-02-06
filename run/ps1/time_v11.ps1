
conda activate tf211pyWin

cd run\fdtd

echo ' BATCH:5 , FUTURE:50 , CONTEXT:10 '
python predict.py     -data type1 -net v11 -t 50  -sp 10 -ex # 123.37 ms

echo ' BATCH:10, FUTURE:25 , CONTEXT:10 '
python predict.py     -data type1 -net v11 -t 25  -sp 10 -ex # 63.33 ms

echo ' BATCH:25, FUTURE:10 , CONTEXT:5 '
python predict.py     -data type1 -net v11 -t 10  -sp 5  -ex # 25.30 ms

cd ..\..\

conda deactivate tf211pyWin

