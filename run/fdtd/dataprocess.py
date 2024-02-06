##############################################
### UNIVERSITY OF NEW MEXICO               ###
### COMPUTATIONAL EM LAB                   ###
### EMULATING FDTD USING TRANSFORMER & GNN ###
### PRE-PROCESSING RAW DATA                ###
### by: OAMEED NOAKOASTEEN                 ###
##############################################

import os
import sys
sys.path.append(os.path.join(sys.path[0],'..','..','lib'))
from paramd import PATHS,FRAMEMNGM,genLIST,trainDIV
from utilsd import getFILENAMES,wHDF
from datad  import genRNDlist,getSCALE,getFIELDS,getBOUND

FRAME         =[FRAMEMNGM[0],FRAMEMNGM[1]]
SHAPE         =[FRAMEMNGM[2],FRAMEMNGM[3]]
RESULTFILENAME=os.path.join(PATHS[2],'results')

#################################
### GENERATE TRAIN/TEST LISTS ###
#################################
if genLIST:
 genRNDlist(PATHS[0],PATHS[2],trainDIV)

####################################
### GET APPROPRIATE SCALE FACTOR ###
####################################
scalefactorE,scalefactorH=getSCALE(PATHS[0],FRAME,SHAPE)
scalefactor              =0.5*(scalefactorE+scalefactorH)
print(' THE MAX OF E-FIELD MAGNITUDE IS '             ,"{0:.4e}".format(scalefactorE),file=open(RESULTFILENAME,'a'))
print(' THE MAX OF H-FIELD MAGNITUDE IS '             ,"{0:.4e}".format(scalefactorH),file=open(RESULTFILENAME,'a'))
print(' THE SCALE FACTOR FOR ALL FIELD COMPONENTS IS ',"{0:.4e}".format(scalefactor ),file=open(RESULTFILENAME,'a'))

########################
### PROCESS RAW DATA ###
########################
filenames=getFILENAMES(PATHS[0])
for i in range(len(filenames)):
 img,maximum_e,maximum_h=getFIELDS(filenames[i],FRAME,SHAPE,scalefactor)
 bnd                    =getBOUND(filenames[i],FRAME,SHAPE)
 savefilename           =os.path.join(PATHS[1],filenames[i].split('\\')[6])
 wHDF(savefilename ,
      ['img','bnd'],
      [ img , bnd ] )
 print(' PROCESSED '   ,filenames[i].split('\\')[6] ,
       ' MAX MAG E IS ',"{0:.4e}".format(maximum_e),
       ' MAX MAG H IS ',"{0:.4e}".format(maximum_h),
       file=open(RESULTFILENAME,'a'                 ))
  

