##############################################
### UNIVERSITY OF NEW MEXICO               ###
### COMPUTATIONAL EM LAB                   ###
### EMULATING FDTD USING TRANSFORMER & GNN ###
### UTILITIES FUNCTION DEFINITIONS         ###
### by: OAMEED NOAKOASTEEN                 ###
##############################################

import os
import numpy      as np
import tensorflow as tf
import h5py

def getFILENAMES(PATH,COMPLETE=True):
 filename=[]
 for FILE in os.listdir(PATH):
  if not FILE.startswith('.'):
   if COMPLETE:
    filename.append(os.path.join(PATH,FILE)) 
   else:
    filename.append(FILE)
 return filename

def sortFILENAMES(FILENAMES,EXPODDM): 
 filenames=[]
 if not EXPODDM:
  sortlist =np.array([])  
  for i in range(len(FILENAMES)):
   filename=FILENAMES[i]
   sortlist=np.append(sortlist,int(filename.split('_')[1].split('.')[0]))
  sortlist =np.sort(sortlist.astype('int'))
  for i in range(len(FILENAMES)):
   filenames.append('simulation'+'_'+str(sortlist[i])+'.h5')
 else:
  sortlist=[]
  for i in range(len(FILENAMES)):
   filename=FILENAMES[i]
   sortlist.append([int(filename.split('_')[1]),int(filename.split('_')[2].split('.')[0])])
  sortlist.sort()
  for i in range(len(FILENAMES)):
   filenames.append('simulation'+'_'+str(sortlist[i][0])+'_'+str(sortlist[i][1])+'.h5')
 return filenames

def wFILE(SAVEPATH,FILENAME,DATA):
 with open(os.path.join(SAVEPATH,FILENAME),'w') as file:
  for i in range(len(DATA)):
   file.write(DATA[i]+'\n')

def rFILE(FILENAME):
 namelist=[]
 with open(FILENAME) as file:
  for line in file:
   namelist.append(line.strip('\n'))
 return namelist

def rHDF(FILENAME):
 fobj=h5py.File(FILENAME,'r')
 keys=[key for key in fobj.keys()]
 return fobj,keys

def wHDF(FILENAME,DIRNAMES,DATA):
 fobj=h5py.File(FILENAME,'w')
 for i in range(len(DIRNAMES)):
  fobj.create_dataset(DIRNAMES[i],data=DATA[i])
 fobj.close()

def plotter(MODE,DATA,CONFIG):
 from matplotlib import pyplot as plt
 eta     =np.sqrt((4*np.pi*1e-7)/(8.854e-12))
 power   =lambda X: 0.5*np.abs(X[:,:,2]/eta)*np.sqrt(np.power(X[:,:,0],2)+np.power(X[:,:,1],2))
 if  MODE in ['monitor']:
  size   =10
  true   =DATA  [0]
  pred   =DATA  [1]
  shape  =true.shape
  rng    =np.random.default_rng(2022)
  indeces=rng.choice(shape[0]*shape[1],size=size,replace=False).tolist()
  idxsort=np.argsort(indeces).tolist()
  indeces=[indeces[i] for i in idxsort]
  true   =np.reshape(true,[-1,shape[2],shape[3],shape[4]])[indeces]
  pred   =np.reshape(pred,[-1,shape[2],shape[3],shape[4]])[indeces]
  fig    =plt.figure(figsize=(size*10,size*10))
  for i in range(size):
   ax    =plt.subplot(2,size,i+1)
   ax.imshow(power(true[i]),cmap='viridis')
   ax.axis('off')
   ax    =plt.subplot(2,size,i+1+size)
   ax.imshow(power(pred[i]),cmap='gray'   )
   ax.axis('off')
  plt.tight_layout()
 else:
  if MODE in ['metrics']:
   label=DATA[0][0]
   data =np.array(DATA[1][0])
   axis =np.array([i/1e3 for i in range(data.shape[0])])
   fig  =plt.figure()
   ax   =plt.subplot()
   ax.plot(axis,data,CONFIG[0][0],linewidth=2,label=label)
   ax.set_title (' Training Metrics ')
   ax.set_xlabel(' Iterations (K) '  )
   ax.legend    ()
   ax.grid      ()
  else:
   if  MODE in ['monitor_gnn']:
    true   =DATA  [0]
    pred   =DATA  [1]
    shape  =true.shape
    true   =np.reshape(true,[-1,shape[2],shape[3],shape[4]])
    pred   =np.reshape(pred,[-1,shape[2],shape[3],shape[4]])
    size   =true.shape[0]    
    fig    =plt.figure(figsize=(size,size))
    for i in range(size):
     ax    =plt.subplot(2,size,i+1)
     ax.imshow(power(true[i]),cmap='viridis')
     ax.axis('off')
     ax    =plt.subplot(2,size,i+1+size)
     ax.imshow(power(pred[i]),cmap='gray'   )
     ax.axis('off')
    plt.tight_layout() 
 return fig

def plot_to_image(figure):
 import                               io
 import                 tensorflow as tf
 from matplotlib import pyplot     as plt
 buf  =io.BytesIO()
 plt.savefig(buf, format='png')
 plt.close  (figure) 
 buf.seek   (0)
 image=tf.image.decode_png(buf.getvalue(), channels=0)
 image=tf.expand_dims     (image         ,          0)
 return image

def check_for_NaNs_and_Infs(DATA,FILENAME):
 filename=FILENAME.split('.')[0]
 NAN_FLAG=np.any(np.isnan(DATA))
 INF_FLAG=np.any(np.isinf(DATA))
 if  NAN_FLAG:
  print (' GENERATED VIDEO FOR '+filename+' CONTAINS NaNs ! ')
 else:
  if INF_FLAG:
   print(' GENERATED VIDEO FOR '+filename+' CONTAINS Infs ! ')
  else:
   print(' GENERATED VIDEO FOR '+filename+' IS OK ')

