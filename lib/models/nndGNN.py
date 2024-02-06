############################################
### UNIVERSITY OF NEW MEXICO             ###
### COMPUTATIONAL EM LAB                 ###
### EMULATING FDTD USING A GNN           ###
### LAYERS, MODEL & CALLBACK DEFINITIONS ###
### by: OAMEED NOAKOASTEEN               ###
############################################

import os
import numpy                  as  np
import tensorflow             as  tf
from matplotlib import pyplot as plt

##############
### LAYERS ###
##############

def create_ffn(hidden_units, dropout_rate, name=None):
 initializer=tf.keras.initializers.he_normal()
 fnn_layers = []
 for units in hidden_units:
  fnn_layers.append(tf.keras.layers.Dense             (units, kernel_initializer=initializer))
  fnn_layers.append(tf.keras.layers.BatchNormalization()                                     )
  fnn_layers.append(tf.keras.layers.ReLU              ()                                     )
 return tf.keras.Sequential(fnn_layers, name=name)

#############
### MODEL ###
#############

class GraphConvLayer(tf.keras.layers.Layer):
    
    def __init__(self                     , 
                 hidden_units             ,
                 create_ffn               ,
                 dropout_rate    =0.2     ,
                 aggregation_type="mean"  ,
                 combination_type="concat",
                 normalize       =False   ,
                 *args                    ,
                 **kwargs                 ,
                                           ):
     super().__init__()
     self.create_ffn       =create_ffn
     self.aggregation_type =aggregation_type
     self.combination_type =combination_type
     self.normalize        =normalize
     self.ffn_prepare      =self.create_ffn(hidden_units, dropout_rate)
     if self.combination_type == "gru":                                                
      self.update_fn       =tf.keras.layers.GRU(units               =hidden_units[0],
                                                activation          ="tanh"         ,
                                                recurrent_activation="sigmoid"      ,
                                                dropout             =dropout_rate   ,
                                                return_sequences    =True           ,  
                                                recurrent_dropout   =0               ) 
     else:
      self.update_fn       =self.create_ffn(hidden_units, dropout_rate)
    
    def prepare(self, node_repesentations, weights=None):
     messages =self.ffn_prepare(node_repesentations)
     if weights is not None:
      messages=messages*tf.expand_dims(weights, -1)
     return messages

    def aggregate(self, node_indices, neighbour_messages):
     num_nodes=tf.math.reduce_max(node_indices) + 1
     if   self.aggregation_type=="sum" :
      aggregated_message  =tf.math.unsorted_segment_sum (neighbour_messages, node_indices, num_segments=num_nodes)
     else:
      if  self.aggregation_type=="mean":
       aggregated_message =tf.math.unsorted_segment_mean(neighbour_messages, node_indices, num_segments=num_nodes)
      else:
       if self.aggregation_type=="max" :
        aggregated_message=tf.math.unsorted_segment_max (neighbour_messages, node_indices, num_segments=num_nodes)
       else:
        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")
     return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
     if   self.combination_type=="gru"   :
      h   =tf.stack ([node_repesentations, aggregated_messages], axis=1)
     else:
      if  self.combination_type=="concat":
       h  =tf.concat([node_repesentations, aggregated_messages], axis=1)
      else:
       if self.combination_type=="add"   :
         h=node_repesentations + aggregated_messages
       else:
        raise ValueError(f"Invalid combination type: {self.combination_type}.")
     node_embeddings =self.update_fn(h)
     if self.combination_type == "gru":
      node_embeddings=tf.unstack        (node_embeddings, axis= 1)[-1]
     if self.normalize:
      node_embeddings=tf.nn.l2_normalize(node_embeddings, axis=-1)
     return node_embeddings

    def call(self, node_repesentations,edges,edge_weights):
     node_indices                            =edges[0] 
     neighbour_indices                       =edges[1]
     neighbour_repesentations                =tf.gather     (node_repesentations     , neighbour_indices )
     neighbour_messages                      =self.prepare  (neighbour_repesentations, edge_weights      )    
     aggregated_messages                     =self.aggregate(node_indices            , neighbour_messages)
     return self.update(node_repesentations, aggregated_messages)

class GraphNN(tf.keras.Model):
    
    def __init__(self                     ,
                 graph_info               ,
                 hidden_units             ,
                 create_ffn               ,
                 GraphConvLayer           ,
                 paths                    ,
                 params                   ,
                 aggregation_type="sum"   ,
                 combination_type="gru"   , 
                 dropout_rate    =0.2     ,
                 normalize       =True    ,
                 *args                    ,
                 **kwargs                  ): 
     super().__init__()
     self.paths         =paths
     self.params        =params
     self.create_ffn    =create_ffn
     self.GraphConvLayer=GraphConvLayer
     self.edges         =graph_info.astype('int32').T
     self.edge_weights  =tf.ones(shape=self.edges.shape[1])
     self.edge_weights  =self.edge_weights/tf.math.reduce_sum(self.edge_weights)
     self.preprocess    =self.create_ffn      (hidden_units      ,  
                                               dropout_rate      , 
                                               name="preprocess"  )
     self.conv1         =self.GraphConvLayer  (hidden_units      ,
                                               self.create_ffn   ,
                                               dropout_rate      ,
                                               aggregation_type  ,
                                               combination_type  ,
                                               normalize         ,
                                               name="graph_conv1" )
     self.postprocess   =self.create_ffn      (hidden_units      , 
                                               dropout_rate      , 
                                               name="postprocess" )
     self.layer_output  =tf.keras.layers.Dense(units=self.params[2][2],
                                               name ="output"          )
     
    def compile(self,optimizer,loss):
     super().compile()
     self.optimizer=optimizer
     self.loss     =loss
    
    def call(self, x):
     shape=x.shape.as_list()
     x    =tf.reshape(x,[-1,np.prod(shape[2:4]),shape[4]])
     x    =self.preprocess  (x)     
     x    =tf.map_fn        (lambda x: self.conv1(x, self.edges, self.edge_weights),x)
     x    =self.postprocess (x)
     x    =self.layer_output(x)
     x    =tf.reshape(x,[-1,*shape[1:]])
     return x
    
    def train_step(self,data):
     fields,objects      =data     
     x                   =fields[:, :-1]
     labels              =fields[:,1:  ]
     with tf.GradientTape() as tape:
      predictions        =self         (x          , training=True  )
      loss               =self.loss    (labels       , predictions            )
     gradients           =tape.gradient(loss         , self.trainable_weights )
     self.optimizer.apply_gradients    (zip(gradients, self.trainable_weights))
     return {"loss":loss}

def loss_function(true,pred):
 def loss_Frob(TRUE,PRED):
  x=tf.subtract   (TRUE, PRED          )
  x=tf.square     (x                   )
  x=tf.reduce_sum (x   , axis=[1,2,3,4])
  x=tf.multiply   (x   , 0.5           )
  x=tf.sqrt       (x                   )
  x=tf.reduce_mean(x                   )
  return x 
 def loss_GDL(YTRUE,YPRED):
  def GRAD(X):
   x=tf.square(X)
   x=tf.reduce_sum(x,axis=[1,2,3,4])
   x=tf.multiply(x,0.5)
   x=tf.reduce_mean(x)
   return x
  shape                =YTRUE.get_shape().as_list() 
  YTRUE_x_shifted_right=tf.slice(YTRUE,[0,0,1,0,0],[shape[0],shape[1],shape[2]-1,shape[3]  ,shape[4]])
  YTRUE_x_shifted_left =tf.slice(YTRUE,[0,0,0,0,0],[shape[0],shape[1],shape[2]-1,shape[3]  ,shape[4]])
  YTRUE_x_GRAD         =tf.abs  (YTRUE_x_shifted_right-YTRUE_x_shifted_left                          )
  YPRED_x_shifted_right=tf.slice(YPRED,[0,0,1,0,0],[shape[0],shape[1],shape[2]-1,shape[3]  ,shape[4]])
  YPRED_x_shifted_left =tf.slice(YPRED,[0,0,0,0,0],[shape[0],shape[1],shape[2]-1,shape[3]  ,shape[4]])
  YPRED_x_GRAD         =tf.abs  (YPRED_x_shifted_right-YPRED_x_shifted_left                          )
  LOSS_x_GRAD          =GRAD    (YPRED_x_GRAD-YTRUE_x_GRAD                                           )
  YTURE_y_shifted_right=tf.slice(YTRUE,[0,0,0,1,0],[shape[0],shape[1],shape[2]  ,shape[3]-1,shape[4]])
  YTURE_y_shifted_left =tf.slice(YTRUE,[0,0,0,0,0],[shape[0],shape[1],shape[2]  ,shape[3]-1,shape[4]])
  YTRUE_y_GRAD         =tf.abs  (YTURE_y_shifted_right-YTURE_y_shifted_left                          )
  YPRED_y_shifted_right=tf.slice(YPRED,[0,0,0,1,0],[shape[0],shape[1],shape[2]  ,shape[3]-1,shape[4]])
  YPRED_y_shifted_left =tf.slice(YPRED,[0,0,0,0,0],[shape[0],shape[1],shape[2]  ,shape[3]-1,shape[4]])
  YPRED_y_GRAD         =tf.abs  (YPRED_y_shifted_right-YPRED_y_shifted_left                          )
  LOSS_y_GRAD          =GRAD    (YPRED_y_GRAD-YTRUE_y_GRAD                                           )
  LOSS_GDL             =LOSS_x_GRAD+LOSS_y_GRAD
  return LOSS_GDL
 loss =loss_Frob(true,pred)+4e-3*loss_GDL(true,pred)
 return loss

##########################
### TRAINING CALLBACKS ###
##########################

class callback_custom_ckpt(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        tf.keras.models.save_model(self.model ,self.model.paths[5])

class callback_custom_monitor(tf.keras.callbacks.Callback):

    def __init__(self, writer, data, plotter, converter):
        super().__init__()
        self.writer   =writer
        self.data     =data
        self.plotter  =plotter
        self.converter=converter

    def on_epoch_begin(self, epoch, logs=None):
        x            =self.data[:, :-1]
        labels       =self.data[:,1:  ]
        predictions  =self.model(x, training=False)        
        figure       =self.plotter('monitor',[labels,x],[])
        image        =self.converter(figure)
        with self.writer.as_default():
         tf.summary.image("Reconstructed Frames", image, step=epoch)

class callback_custom_metrics_batch(tf.keras.callbacks.Callback):
    
    def __init__(self, writer):
        super().__init__()
        self.writer   =writer
    
    def on_train_batch_end(self, batch, logs=None):
        with self.writer.as_default():
         tf.summary.scalar('batch_loss', logs['loss'], step=batch)

class callback_custom_history(tf.keras.callbacks.Callback):
    
    def __init__(self, plotter, whdf):
        super().__init__()
        self.plotter=plotter
        self.whdf   =whdf
    
    def on_train_begin(self, logs=None):
        self.metric_key  =[]
        self.metric_one  =[]

    def on_train_batch_end(self, batch, logs=None):     
        keys=list(logs.keys())
        self.metric_one.append  (logs[keys[0]])
        if batch==0:
         self.metric_key.append (     keys[0] )
    
    def on_train_end(self, logs=None):
        savefilename=os.path.join(self.model.paths[6],'train','metrics')
        self.whdf(savefilename+'.h5'  ,
                  [self.metric_key[0]],
                  [self.metric_one   ] )
        fig=self.plotter('metrics'              ,
                         [[self.metric_key[0]],
                          [self.metric_one   ] ],
                         [['b'               ] ] )
        plt.savefig(savefilename+'.png',format='png')

