####################################################
### UNIVERSITY OF NEW MEXICO                     ###
### COMPUTATIONAL EM LAB                         ###
### EMULATING FDTD USING TRANSFORMER & GNN       ###
### GENERATING PREDICTIONS USING A TRAINED MODEL ###
### by: OAMEED NOAKOASTEEN                       ###
####################################################

import                              os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import                              sys
sys.path.append(os.path.join(sys.path[0],'..','..','lib'         ))
sys.path.append(os.path.join(sys.path[0],'..','..','lib','models'))
import numpy                    as  np
import tensorflow               as  tf
from   matplotlib import pyplot as  plt
from   paramd     import            PATHS, PARAMS
from   utilsd     import           (rFILE                  ,
                                    rHDF                   ,
                                    check_for_NaNs_and_Infs,
                                    wHDF                    )
from   inputd     import            get_data
import time     

if  PARAMS[2][0] in ['v11','v12','v13']:
 from nndTRNSF import (CustomSchedule,
                       Propagator     )
 custom_object_dict                ={"CustomSchedule":CustomSchedule}
 model                             =tf.keras.models.load_model(PATHS[5], custom_objects=custom_object_dict) 
 propagator                        =Propagator                (model   , PARAMS[2][1]-PARAMS[5][9]        )
 filenames                         =[(fn,os.path.join(PATHS[1],fn)) for fn in rFILE(os.path.join(PATHS[2],'listTEST'))]
 for fn, full_fn in filenames:
  savefilename                     =os.path.join(PATHS[7], fn     )
  fields, objects                  =get_data    ( PARAMS , full_fn)
  fields_past                      =fields      [:,:PARAMS[5][9]  ]
  fields_future                    =fields      [:, PARAMS[5][9]: ]
  fields_future_initial_frame      =fields_past [:,-1:]
  
  fields_past_shape                =fields_past.shape
  fields_future_initial_frame_shape=fields_future_initial_frame.shape  
  fields_past                      =np.reshape(fields_past                ,[*fields_past_shape                [:2],np.prod(fields_past_shape                [2:])])
  fields_future_initial_frame      =np.reshape(fields_future_initial_frame,[*fields_future_initial_frame_shape[:2],np.prod(fields_future_initial_frame_shape[2:])])
  field_future                     =np.reshape(fields_future              ,[-1, *fields_past_shape[2:]])

  fields_future_predicted, tm      =propagator(fields_past,fields_future_initial_frame)
  print("Inference time is {:1.2f} ms".format(tm))
  if PARAMS[8][0]: # FOR TIMING STUDIES
   exit()
  fields_future_predicted_shape    =fields_future_predicted.shape
  fields_future_predicted          =np.reshape(fields_future_predicted    ,[-1, *fields_past_shape[2:]])
  
  check_for_NaNs_and_Infs(fields_future_predicted, fn)
 
  wHDF(savefilename                                                 ,
       ['vid_true'   ,'vid_pred'              ,'bnd'               ],
       [ field_future, fields_future_predicted, np.squeeze(objects)] ) 
 
else:
 if PARAMS[2][0] in ['v21','v22','v23','v25','v26','v27']:  
  model          =tf.keras.models.load_model(PATHS[5]) 
  filenames      =[(fn,os.path.join(PATHS[1],fn)) for fn in rFILE(os.path.join(PATHS[2],'listTEST'))]
  for fn, full_fn in filenames:
   savefilename  =os.path.join(PATHS[7]          , fn            )
   fields,objects=get_data    ( PARAMS           , full_fn       )
   x             =fields[:,:-1]
   tb            =time.time()
   predictions   =model(x)
   te            =time.time()
   tm            =te-tb
   print("Inference time is {:1.2f} ms".format((tm/np.prod(x.shape[:2]))/1e-3)) 
   if PARAMS[8][0]: # FOR TIMING STUDIES
    exit()
   predictions   =np.reshape  (predictions, [-1,PARAMS[0][2],PARAMS[0][3],PARAMS[2][2]])
   x             =np.reshape  (x          , [-1,PARAMS[0][2],PARAMS[0][3],PARAMS[2][2]])
   check_for_NaNs_and_Infs      (predictions, fn    )
   wHDF(savefilename                                  ,
        ['vid_true','vid_pred'  ,           'bnd'    ],
        [ x        , predictions, np.squeeze(objects)] ) 


