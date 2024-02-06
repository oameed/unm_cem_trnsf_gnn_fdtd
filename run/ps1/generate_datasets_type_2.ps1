
conda activate tf211pyWin

cd run\fdtd

echo ' GENERATING DATASET type2'
matlab -batch "generator('type2',false,{})" -noFigureWindows

echo ' PROCESSING DATASET type2'
python dataprocess.py -data type2

cd ..\..\

conda deactivate tf211pyWin

# THE SPEC FILE FOR 'type2' WAS GENERATED USING:
# generator('type2',true,{100,[],linspace(0.4,0.6,3),linspace(0.4,0.6,3),[0.4,0.5,0.6]})
# THE LIST OF FILE NAMES FOR TRAIN/TEST ASSIGNMENTS FOR 'type2' WERE GENERATED USING:
# python dataprocess.py -data type2 -list

