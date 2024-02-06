##########################################
### UNIVERSITY OF NEW MEXICO           ###
### COMPUTATIONAL EM LAB               ###
### EMULATING FDTD USING A TRANSFORMER ###
### TRAINING                           ###
### by: OAMEED NOAKOASTEEN             ###
##########################################

import                             os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import                             sys
sys.path.append(os.path.join(sys.path[0],'..','..','lib'         ))
sys.path.append(os.path.join(sys.path[0],'..','..','lib','models'))
import numpy                    as np
import tensorflow               as tf
from   paramd     import           PATHS, PARAMS 
from   inputd     import          (get_data               ,
                                   saveTFRECORDS          ,
                                   inputTFRECORDS          )
from   utilsd     import          (plotter                ,
                                   plot_to_image          ,
                                   wHDF                    )
from   nndTRNSF   import          (positional_encoding    ,
                                   PositionalEmbedding    ,
                                   CrossAttention         ,
                                   GlobalSelfAttention    ,
                                   CausalSelfAttention    ,
                                   FeedForward            ,
                                   EncoderLayer           ,
                                   DecoderLayer           ,
                                   Encoder                ,
                                   Decoder                ,
                                   Transformer            ,
                                   CustomSchedule         ,
                                   loss_function          ,
                                   callback_custom_ckpt   ,
                                   callback_custom_monitor,
                                   callback_custom_history )

LISTTRAIN      =os.path.join(PATHS[2],'listTRAIN'    )
TFRTRAIN       =os.path.join(PATHS[4],'train'        )
LOGTRAIN       =os.path.join(PATHS[6],'train'        )
monitor_data_fn=os.path.join(PATHS[1],PARAMS[6][0]   )
monitor_data   =get_data    (PARAMS  ,monitor_data_fn)

#############
### MODEL ###
#############
model         =Transformer(num_layers         =PARAMS[5][0]       ,
                           d_model            =PARAMS[5][1]       ,
                           num_heads          =PARAMS[5][3]       ,
                           dff                =PARAMS[5][2]       ,
                           input_vocab_size   =PARAMS[5][5]       ,
                           target_vocab_size  =PARAMS[5][6]       ,
                           POSITIONAL_ENCODING=positional_encoding,
                           POSITIONALEMBEDDING=PositionalEmbedding,
                           CROSSATTENTION     =CrossAttention     ,
                           GLOBALSELFATTENTION=GlobalSelfAttention,
                           CAUSALSELFATTENTION=CausalSelfAttention,
                           FEEDFORWARD        =FeedForward        , 
                           ENCODERLAYER       =EncoderLayer       ,
                           DECODERLAYER       =DecoderLayer       ,
                           ENCODER            =Encoder            ,
                           DECODER            =Decoder            ,
                           paths              =PATHS              ,
                           params             =PARAMS             ,
                           dropout_rate       =PARAMS[5][4]        )

learning_rate =CustomSchedule(PARAMS[5][1])

model.compile(optimizer  =tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
              loss       = loss_function                                                                 )

#################
### CALLBACKS ###
#################
callbacks_train=[callback_custom_ckpt          ()                                                        ,
                 callback_custom_monitor       (writer        =tf.summary.create_file_writer(LOGTRAIN ),
                                                data          =monitor_data                            ,
                                                plotter       =plotter                                 ,
                                                converter     =plot_to_image                            ),                
                 callback_custom_history       (plotter       =plotter                                 ,
                                                whdf          =wHDF                                     ),
                 tf.keras.callbacks.TensorBoard(log_dir       =LOGTRAIN                                , 
                                                histogram_freq=1                                       , 
                                                update_freq   ='batch'                                  ) ]

#############
### TRAIN ###
#############
print        (' WRITING  TRAINING   DATA TO TFRECORDS FORMAT ')
saveTFRECORDS(PATHS, PARAMS, LISTTRAIN    )
print        (' FITTING  MODEL'                               )
data_train          =inputTFRECORDS(TFRTRAIN     ,PARAMS )

fields_past         =monitor_data[0][:,:PARAMS[5][9]     ]
fields_future       =monitor_data[0][:, PARAMS[5][9]-1:-1]
fields_past_shape   =fields_past.shape
fields_future_shape =fields_future.shape
fields_past         =np.reshape(fields_past  ,[*fields_past_shape  [:2],np.prod(fields_past_shape  [2:])])
fields_future       =np.reshape(fields_future,[*fields_future_shape[:2],np.prod(fields_future_shape[2:])])

data_validation    =((fields_past,fields_future),(fields_past,fields_future))

model.fit    (x                    =data_train     ,
              epochs               =PARAMS[3][1]   ,
              validation_data      =data_validation,
              callbacks            =callbacks_train,
              verbose              =1               )
print        (' TRAINING FINISHED '                           )

