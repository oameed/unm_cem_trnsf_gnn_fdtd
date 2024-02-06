
cd run\fdtd

#matlab -batch "graphics('v11',3,'Power',0.5 ,0.001,1  )" -noFigureWindows
#matlab -batch "graphics('v12',3,'Power',0.5 ,0.001,1  )" -noFigureWindows
#matlab -batch "graphics('v13',3,'Power',0.5 ,0.001,1  )" -noFigureWindows

matlab -batch "graphics('v21',3,'Power',0.5 ,0.01 ,0.5)" -noFigureWindows
matlab -batch "graphics('v22',3,'Power',0.5 ,0.01 ,0.5)" -noFigureWindows
matlab -batch "graphics('v23',3,'Power',0.5 ,0.01 ,0.5)" -noFigureWindows

cd ..\..\

