##################################################
### UNIVERSITY OF NEW MEXICO                   ###
### COMPUTATIONAL EM LAB                       ###
### EMULATING FDTD USING GRAPH NEURAL NETWORKS ###
### TRAINING                                   ###
### by: OAMEED NOAKOASTEEN                     ###
##################################################

import               os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import               sys
sys.path.append(os.path.join(sys.path[0],'..','..','lib'         ))
sys.path.append(os.path.join(sys.path[0],'..','..','lib','models'))
import numpy      as np
import tensorflow as tf
from paramd import   PATHS, PARAMS
from inputd import  (get_graph_info               ,
                     get_data                     ,
                     saveTFRECORDS                ,
                     inputTFRECORDS                )
from utilsd import  (plotter                      ,
                     wHDF                         ,
                     plot_to_image                 )

from nndGNN import  (callback_custom_ckpt         , 
                     callback_custom_monitor      ,
                     callback_custom_metrics_batch,
                     callback_custom_history      , 
                     create_ffn                   ,
                     GraphConvLayer               ,
                     GraphNN                      ,
                     loss_function                 )

LISTTRAIN         =os.path.join(PATHS[2],'listTRAIN' )
TFRTRAIN          =os.path.join(PATHS[4],'train'     )
LOGTRAIN          =os.path.join(PATHS[6],'train'     )
monitor_data_fn   =os.path.join(PATHS[1],PARAMS[6][0]   )
monitor_data      =get_data    (PARAMS  ,monitor_data_fn)[0]

graph_info     =get_graph_info(PATHS,PARAMS)

#############
### MODEL ###
#############
model     =GraphNN(graph_info    =graph_info    ,
                   hidden_units  =PARAMS[7][0]  ,
                   create_ffn    =create_ffn    ,
                   GraphConvLayer=GraphConvLayer,
                   paths         =PATHS         ,
                   params        =PARAMS        ,
                   dropout_rate  =PARAMS[7][1]  ,
                   name          ="gnn_model"    )

model.compile     (optimizer     =tf.keras.optimizers.Adam(1e-4),  
                   loss          = loss_function                 )

#################
### CALLBACKS ###
#################
callbacks_train=[callback_custom_ckpt          ()                                                        ,
                 callback_custom_monitor       (writer        =tf.summary.create_file_writer(LOGTRAIN) ,
                                                data          =monitor_data                            ,
                                                plotter       =plotter                                 ,
                                                converter     =plot_to_image                            ),
                 callback_custom_history       (plotter       =plotter                                 ,
                                                whdf          =wHDF                                     ),
                 callback_custom_metrics_batch (writer        =tf.summary.create_file_writer(LOGTRAIN))  ,
                 tf.keras.callbacks.TensorBoard(log_dir       =LOGTRAIN                                , 
                                                histogram_freq=1                                       , 
                                                update_freq   ='batch'                                  ) ]

#############
### TRAIN ###
#############
print        (' WRITING  TRAINING   DATA TO TFRECORDS FORMAT ')
saveTFRECORDS(PATHS, PARAMS, LISTTRAIN                        )
print        (' FITTING  MODEL'                               )
data_train        =inputTFRECORDS(TFRTRAIN     ,PARAMS  )

data_validation   =monitor_data[:,:-1]
data_validation   =(data_validation,data_validation)

model.fit(x                    =data_train     ,
          epochs               =PARAMS[3][1]   ,
          validation_data      =data_validation,
          callbacks            =callbacks_train,
          verbose              =1               )
print    (' TRAINING FINISHED '                 )


