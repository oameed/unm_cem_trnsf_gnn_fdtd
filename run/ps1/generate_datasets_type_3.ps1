
conda activate tf211pyWin

cd run\fdtd

echo ' GENERATING DATASET type3'
matlab -batch "generator('type3',false,{})" -noFigureWindows

echo ' PROCESSING DATASET type3'
python dataprocess.py -data type3

cd ..\..\

conda deactivate tf211pyWin

# THE SPEC FILE FOR 'type3' WAS GENERATED USING:
# generator('type3',true,{100,[],linspace(0.4,0.6,3),[],[]})
# THE LIST OF FILE NAMES FOR TRAIN/TEST ASSIGNMENTS FOR 'type3' WERE GENERATED USING:
# python dataprocess.py -data type3 -list

