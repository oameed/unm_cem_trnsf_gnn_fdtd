
conda activate tf211pyWin

cd run\fdtd

echo ' GENERATING DATASET type1'
matlab -batch "generator('type1',false,{})" -noFigureWindows

echo ' PROCESSING DATASET type1'
python dataprocess.py -data type1 -simb 150

cd ..\..\

conda deactivate tf211pyWin

# THE SPEC FILE FOR 'type1' WAS GENERATED USING:
# generator('type1',true,{100,linspace(2*pi/16,6*pi/16,5),[],linspace(0.4,0.6,3),[0.4,0.5,0.6]})
# THE LIST OF FILE NAMES FOR TRAIN/TEST ASSIGNMENTS FOR 'type1' WERE GENERATED USING:
# python dataprocess.py -data type1 -simb 150 -list
