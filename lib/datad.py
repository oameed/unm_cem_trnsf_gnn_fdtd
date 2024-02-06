##############################################
### UNIVERSITY OF NEW MEXICO               ###
### COMPUTATIONAL EM LAB                   ###
### EMULATING FDTD USING TRANSFORMER & GNN ###
### DATA PROCESSING FUNCTION DEFINITIONS   ###
### by: OAMEED NOAKOASTEEN                 ###
##############################################

import                                 os
import              numpy           as np
from utilsd import (getFILENAMES ,
                    sortFILENAMES,
                    wFILE        ,
                    rHDF          )

def genRNDlist(READPATH,SAVEPATH,DIV,EXPODDM=False):
 filenames=getFILENAMES(READPATH,False)
 np.random.shuffle(filenames) 
 trainlist=filenames[0:int(np.floor(DIV*len(filenames))) ]
 testlist =filenames[  int(np.floor(DIV*len(filenames))):]
 trainlist=sortFILENAMES(trainlist,EXPODDM)
 testlist =sortFILENAMES(testlist ,EXPODDM)
 wFILE(SAVEPATH,'listTRAIN',trainlist)
 wFILE(SAVEPATH,'listTEST' ,testlist )

def trim(DATA,SHAPE):
 difw=DATA.shape[1]-SHAPE[0]
 difh=DATA.shape[2]-SHAPE[1]
 if np.mod(difw,2)==0:
  skipwb=int(difw/2            )
  skipwe=int(difw/2            )
 else:
  skipwb=int(np.floor(difw/2)  )
  skipwe=int(np.floor(difw/2)+1)
 if np.mod(difh,2)==0:
  skiphb=int(difh/2            )
  skiphe=int(difh/2            )
 else:
  skiphb=int(np.floor(difh/2)  )
  skiphe=int(np.floor(difh/2)+1)
 DATA=DATA[:,skipwb:DATA.shape[1]-skipwe,skiphb:DATA.shape[2]-skiphe]
 return DATA

def getSCALE(HDFDATAPATH,FRAME,SHAPE):
 eta      =np.sqrt((4*np.pi*1e-7)/(8.85e-12)) 
 mag_max_E=np.array([])                                                                  # FIND THE MAX MAGNITUDE OF E-FIELD VECTOR / H-FIELD VECTOR 
 mag_max_H=np.array([])
 filenames=getFILENAMES(HDFDATAPATH)
 for i in range(len(filenames)):
  fobj,keys=rHDF(filenames[i])
  imgex       =trim(fobj['Ex_FIELD'][FRAME[0]-1:FRAME[1]-1], [SHAPE[0],SHAPE[1]]) 
  imgey       =trim(fobj['Ey_FIELD'][FRAME[0]-1:FRAME[1]-1], [SHAPE[0],SHAPE[1]])
  imghz       =trim(fobj['Hz_FIELD'][FRAME[0]-1:FRAME[1]-1], [SHAPE[0],SHAPE[1]])*eta
  maximum_E   =np.amax(np.sqrt(np.power(imgex,2)+np.power(imgey,2)))
  maximum_H   =np.amax(np.absolute(imghz))   
  mag_max_E   =np.append(mag_max_E,maximum_E)
  mag_max_H   =np.append(mag_max_H,maximum_H)
 scalefactor_E=np.amax(mag_max_E)
 scalefactor_H=np.amax(mag_max_H)
 return scalefactor_E,scalefactor_H

def normalize(EX,EY,HZ,SCALEFACTOR):
 for i in range(EX.shape[0]):
  EX[i]   =EX[i]/SCALEFACTOR
  EY[i]   =EY[i]/SCALEFACTOR
  HZ[i]   =HZ[i]/SCALEFACTOR
 maximum_e=np.amax(np.sqrt(np.power(EX,2)+np.power(EY,2)))
 maximum_h=np.amax(np.absolute(HZ))
 return EX,EY,HZ,maximum_e,maximum_h

def getFIELDS(FILENAME,FRAME,SHAPE,SCALEFACTOR):
 eta=np.sqrt((4*np.pi*1e-7)/(8.85e-12))
 # FIELD DATA: SELECT EVENTFUL TIME BLOCK
 # FIELD DATA: TRIM TO REQUIRED PROJECT DIMENSIONS
 # FIELD DATA: SCALE Hz COMPONENT WITH ETA (FREE-SPACE IMPEDANCE)
 # FIELD DATA: NORMALIZE EACH TIME-FRAME WITH SCALEFACTOR
 # FIELD DATA: STACK ALL FIELD COMPONENTS TO CREATE UNIFIED DATA STRUCTURE
 # FIELD DATA: SET THE DATA TYPE TO 'float32'
 fobj,keys=rHDF(FILENAME)
 imgex=trim(fobj['Ex_FIELD'][FRAME[0]-1:FRAME[1]-1], [SHAPE[0],SHAPE[1]]) 
 imgey=trim(fobj['Ey_FIELD'][FRAME[0]-1:FRAME[1]-1], [SHAPE[0],SHAPE[1]])
 imghz=trim(fobj['Hz_FIELD'][FRAME[0]-1:FRAME[1]-1], [SHAPE[0],SHAPE[1]])*eta
 imgex,imgey,imghz,maximum_e,maximum_h=normalize(imgex,imgey,imghz,SCALEFACTOR)
 img  =np.stack((imgex,imgey,imghz),axis=3).astype('float32')
 return img,maximum_e,maximum_h

def getBOUND(FILENAME,FRAME,SHAPE):
 # BOUNDARY DATA: CONVERT TO BINARY ('1':PEC,'0':FREE-SPACE)
 # BOUNDARY DATA: MAKE AN ARRAY OUT OF BOUNDARY THE SAME NUMBER OF TIME-FRAMES AS FIELDS
 # BOUNDARY DATA: TRIM TO REQUIRED PROJECT DIMENSIONS
 # BOUNDARY DATA: RESHAPE TO CONFORM TO HAVE A '1 CHANNEL' FORMAT
 # BOUNDARY DATA: SET THE DATA TYPE TO 'float32'
 fobj,keys=rHDF(FILENAME)
 shape    =fobj['boundary'].shape
 bnd      =np.zeros((shape[0],shape[1]))
 for  indx1 in range(shape[0]):
  for indx2 in range(shape[1]):
   if not fobj['boundary'][indx1,indx2]==0:
    bnd[indx1,indx2]=1
 bnd      =np.stack([ bnd for _ in range(FRAME[1]-FRAME[0])])
 bnd      =trim(bnd,[SHAPE[0],SHAPE[1]])
 bnd      =np.expand_dims(bnd,axis=-1).astype('float32')
 return bnd


