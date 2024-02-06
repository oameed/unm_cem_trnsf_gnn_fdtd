############################################
### UNIVERSITY OF NEW MEXICO             ###
### COMPUTATIONAL EM LAB                 ###
### EMULATING FDTD USING A TRANSFORMER   ###
### LAYERS, MODEL & CALLBACK DEFINITIONS ###
### by: OAMEED NOAKOASTEEN               ###
############################################

import os
import numpy                  as  np
import tensorflow             as  tf
from matplotlib import pyplot as plt
import time

def positional_encoding(length, depth):
 depth       = depth/2
 positions   = np.arange(length)[:, np.newaxis]     
 depths      = np.arange(depth)[np.newaxis, :]/depth
 angle_rates = 1 / (10000**depths)         
 angle_rads  = positions * angle_rates     
 pos_encoding= np.concatenate([np.sin(angle_rads), np.cos(angle_rads)],axis=-1)
 return tf.cast(pos_encoding, dtype=tf.float32)

##############
### LAYERS ###
##############
class PositionalEmbedding(tf.keras.layers.Layer):
    
    def __init__(self               ,
                 vocab_size         ,
                 d_model            ,
                 POSITIONAL_ENCODING ):
     super().__init__()
     self.d_model     = d_model     
     self.dense       = tf.keras.layers.Dense    (d_model                             )
     self.pos_encoding= POSITIONAL_ENCODING      (length=2048, depth=d_model          )
     
    def call(self, x):     
     length = tf.shape(x)[1]
     x      = self.dense(x)
     x     *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
     x      = x + self.pos_encoding[tf.newaxis, :length, :]
     return x

class BaseAttention(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
     super().__init__()
     self.mha             = tf.keras.layers.MultiHeadAttention(**kwargs)
     self.layernorm       = tf.keras.layers.LayerNormalization()
     self.add             = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
    
    def call(self, x, context):
     attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)
     x                        = self.add([x, attn_output])
     x                        = self.layernorm(x)
     return x

class GlobalSelfAttention(BaseAttention):
    
    def call(self, x):
     attn_output = self.mha(query=x, value=x, key=x)
     x           = self.add([x, attn_output])
     x           = self.layernorm(x)
     return x

class CausalSelfAttention(BaseAttention):
    
    def call(self, x):
     attn_output = self.mha(query=x, value=x, key=x, use_causal_mask = True)
     x           = self.add([x, attn_output])
     x           = self.layernorm(x)
     return x

class FeedForward(tf.keras.layers.Layer):
    
    def __init__(self            ,
                 d_model         ,
                 dff             ,
                 dropout_rate=0.1 ):
     super().__init__()
     self.seq        = tf.keras.Sequential([tf.keras.layers.Dense  (dff         , activation='relu'),
                                            tf.keras.layers.Dense  (d_model                        ),
                                            tf.keras.layers.Dropout(dropout_rate                   ) ])
     self.add        = tf.keras.layers.Add()
     self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
     x = self.add       ([x, self.seq(x)])
     x = self.layer_norm(x) 
     return x

class EncoderLayer(tf.keras.layers.Layer):
    
    def __init__(self               ,
                 *                  ,
                 d_model            ,
                 num_heads          ,
                 dff                ,
                 GLOBALSELFATTENTION,
                 FEEDFORWARD        ,
                 dropout_rate=0.1    ):
     super().__init__()
     self.self_attention = GLOBALSELFATTENTION(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate) 
     self.ffn            = FEEDFORWARD          (  d_model, dff)
    
    def call(self, x):
     x               = self.self_attention(x)
     x               = self.ffn           (x)
     return x

class DecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self               ,
                 *                  ,
                 d_model            ,
                 num_heads          ,
                 dff                ,
                 CAUSALSELFATTENTION,
                 CROSSATTENTION     ,
                 FEEDFORWARD        ,
                 dropout_rate=0.1    ):
     super().__init__()
     self.causal_self_attention = CAUSALSELFATTENTION(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate) 
     self.cross_attention       = CROSSATTENTION     (num_heads=num_heads, key_dim=d_model, dropout=dropout_rate) 
     self.ffn                   = FEEDFORWARD        (d_model, dff)

    def call(self, x, context):
     x                     = self.causal_self_attention(x=x)
     x                     = self.cross_attention      (x=x, context=context)
     x                     = self.ffn                  (x)
     return x

#############
### MODEL ###
#############
class Encoder(tf.keras.layers.Layer):
    
    def __init__(self               ,
                 *                  ,
                 num_layers         ,
                 d_model            ,
                 num_heads          ,
                 dff                , 
                 vocab_size         ,
                 POSITIONAL_ENCODING,
                 POSITIONALEMBEDDING,
                 GLOBALSELFATTENTION,
                 FEEDFORWARD        ,
                 ENCODERLAYER       ,
                 dropout_rate=0.1    ):
     super().__init__()
     self.d_model      = d_model
     self.num_layers   = num_layers
     self.pos_embedding= POSITIONALEMBEDDING(vocab_size         =vocab_size         ,
                                             d_model            =d_model            ,
                                             POSITIONAL_ENCODING=POSITIONAL_ENCODING )                      
     self.enc_layers   = [ENCODERLAYER(d_model            =d_model            , 
                                       num_heads          =num_heads          , 
                                       dff                =dff                ,
                                       GLOBALSELFATTENTION=GLOBALSELFATTENTION,
                                       FEEDFORWARD        =FEEDFORWARD        ,
                                       dropout_rate       =dropout_rate        ) for _ in range(num_layers)] 
     self.dropout      = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
     x  = self.pos_embedding(x)
     x  = self.dropout(x)
     for i in range(self.num_layers):
      x = self.enc_layers[i](x)
     return x                    

class Decoder(tf.keras.layers.Layer):
    
    def __init__(self               ,
                 *                  ,
                 num_layers         ,
                 d_model            ,
                 num_heads          ,
                 dff                ,
                 vocab_size         ,
                 POSITIONAL_ENCODING,
                 POSITIONALEMBEDDING,
                 CAUSALSELFATTENTION,
                 CROSSATTENTION     ,
                 FEEDFORWARD        ,
                 DECODERLAYER       ,
                 dropout_rate=0.1    ):
     super().__init__()
     self.d_model          = d_model
     self.num_layers       = num_layers
     self.pos_embedding    = POSITIONALEMBEDDING(vocab_size         =vocab_size         ,
                                                 d_model            =d_model            ,
                                                 POSITIONAL_ENCODING=POSITIONAL_ENCODING )
     self.dropout          = tf.keras.layers.Dropout(dropout_rate)
     self.dec_layers       = [DECODERLAYER(d_model            =d_model            , 
                                           num_heads          =num_heads          , 
                                           dff                =dff                , 
                                           CAUSALSELFATTENTION=CAUSALSELFATTENTION,
                                           CROSSATTENTION     =CROSSATTENTION     ,
                                           FEEDFORWARD        =FEEDFORWARD        ,
                                           dropout_rate       =dropout_rate        ) for _ in range(num_layers)]
    
    def call(self, x, context):
     x = self.pos_embedding(x)
     x = self.dropout      (x)
     for i in range(self.num_layers):
      x = self.dec_layers[i](x, context)
     return x

class Transformer(tf.keras.Model):
    
    def __init__(self               ,
                 *                  ,
                 num_layers         ,  
                 d_model            ,
                 num_heads          ,
                 dff                , 
                 input_vocab_size   ,  
                 target_vocab_size  ,
                 POSITIONAL_ENCODING,
                 POSITIONALEMBEDDING,
                 CROSSATTENTION     ,
                 GLOBALSELFATTENTION,
                 CAUSALSELFATTENTION,
                 FEEDFORWARD        , 
                 ENCODERLAYER       ,
                 DECODERLAYER       ,
                 ENCODER            ,
                 DECODER            ,
                 paths              ,
                 params             ,
                 dropout_rate=0.1    ):
     super().__init__()
     self.encoder     = ENCODER(num_layers         =num_layers         ,  
                                d_model            =d_model            ,
                                num_heads          =num_heads          ,  
                                dff                =dff                ,
                                vocab_size         =input_vocab_size   ,
                                POSITIONAL_ENCODING=POSITIONAL_ENCODING,
                                POSITIONALEMBEDDING=POSITIONALEMBEDDING,
                                GLOBALSELFATTENTION=GLOBALSELFATTENTION,
                                FEEDFORWARD        =FEEDFORWARD        ,
                                ENCODERLAYER       =ENCODERLAYER       ,
                                dropout_rate       =dropout_rate        ) 

     self.decoder     = DECODER(num_layers         =num_layers         ,   
                                d_model            =d_model            ,
                                num_heads          =num_heads          ,  
                                dff                =dff                ,
                                vocab_size         =target_vocab_size  ,
                                POSITIONAL_ENCODING=POSITIONAL_ENCODING,
                                POSITIONALEMBEDDING=POSITIONALEMBEDDING,
                                CAUSALSELFATTENTION=CAUSALSELFATTENTION,
                                CROSSATTENTION     =CROSSATTENTION     ,
                                FEEDFORWARD        =FEEDFORWARD        ,
                                DECODERLAYER       =DECODERLAYER       ,
                                dropout_rate       =dropout_rate        )     
     self.paths       = paths
     self.params      = params
     self.final_layer = tf.keras.layers.Dense(np.prod([self.params[0][2],self.params[0][3],self.params[2][2]]))
    
    def compile(self, optimizer, loss):
     super().compile()
     self.optimizer=optimizer
     self.loss     =loss
        
    @tf.function(input_signature=[(tf.TensorSpec(shape=(None, None, 49152), dtype=tf.float32), 
                                   tf.TensorSpec(shape=(None, None, 49152), dtype=tf.float32) )])  
    def call(self, inputs):
     #context, x= inputs
     context, x= inputs
     context   = self.encoder    (   context)
     x         = self.decoder    (x, context)
     x         = self.final_layer(x) 
     return x
    
    def train_step(self, data):
     fields, objects     =data
     fields_past         =fields [:,:self.params[5][9]     ] # CONTEXT
     fields_future       =fields [:, self.params[5][9]-1:-1]
     label               =fields [:, self.params[5][9]  :  ]
     
     fields_past_shape   =fields_past.shape.as_list   ()[1:]
     fields_future_shape =fields_future.shape.as_list ()[1:]
     fields_past         =tf.reshape(fields_past  ,[-1,fields_past_shape  [0]]+[np.prod(fields_past_shape  [1:])])
     fields_future       =tf.reshape(fields_future,[-1,fields_future_shape[0]]+[np.prod(fields_future_shape[1:])])
     
     with tf.GradientTape() as tape:
      predictions    =self           ((fields_past,fields_future))
      pred_shape     =predictions.shape.as_list()[1:]
      predictions    =tf.reshape(predictions,[-1,pred_shape[0],*fields_past_shape[1:]])
      loss           =self.loss      (label ,predictions         )
     gradients=tape.gradient       (    loss      , self.trainable_weights )
     self.optimizer.apply_gradients(zip(gradients , self.trainable_weights))
     return {'loss':loss}

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, d_model, warmup_steps=4000):
     super().__init__()
     self.d_model      = d_model
     self.d_model      = tf.cast(self.d_model, tf.float32)
     self.warmup_steps = warmup_steps
    
    def __call__(self, step):
     step = tf.cast      (step, dtype=tf.float32)
     arg1 = tf.math.rsqrt(step)
     arg2 = step * (self.warmup_steps ** -1.5)
     return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
     return {"d_model":self.d_model.numpy(),"warmup_steps":self.warmup_steps}

def loss_function(TRUE,PRED):
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
 loss=loss_Frob(TRUE,PRED)+4e-3*loss_GDL(TRUE,PRED)
 return loss

class Propagator(tf.Module):
    
    def __init__(self, transformer, max_length):
     self.transformer= transformer
     self.max_length = max_length
    
    def __call__(self, fields_past,fields_future_initial_frame):
     time_array      = []
     output_array    = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
     permutation     = [1,0,2]
     output_array    = output_array.write(0, tf.transpose(fields_future_initial_frame, perm=permutation))
     for i in range(self.max_length):
      tb =time.time()
      predictions    = self.transformer((fields_past , tf.transpose(output_array.concat(), perm=permutation)), training=False)
      te =time.time()
      time_array.append((te-tb)/fields_past.shape[0])      
      predictions    = predictions[:,-1:,:].numpy()
      output_array   = output_array.write(i+1, tf.transpose(predictions, perm=permutation))
     time_average    = sum(time_array)/len(time_array)
     time_average    = time_average/1e-3
     output_array    = tf.transpose(output_array.concat(), perm=permutation)
     return output_array.numpy()[:,1:], time_average

##########################
### TRAINING CALLBACKS ###
##########################
class callback_custom_ckpt(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        tf.keras.models.save_model(self.model, self.model.paths[5])

class callback_custom_monitor(tf.keras.callbacks.Callback):

    def __init__(self, writer, data, plotter, converter):
        super().__init__()
        self.writer   =writer
        self.data     =data
        self.plotter  =plotter
        self.converter=converter

    def on_epoch_begin(self, epoch, logs=None):
        fields, objects     =self.data
        fields_past         =fields [:,:self.model.params[5][9]     ] 
        fields_future       =fields [:, self.model.params[5][9]-1:-1]
        label               =fields [:, self.model.params[5][9]  :  ]
               
        fields_past_shape  =fields_past.shape
        fields_future_shape=fields_future.shape        
        fields_past        =np.reshape(fields_past  ,[*fields_past_shape  [:2],np.prod(fields_past_shape  [2:])])
        fields_future      =np.reshape(fields_future,[*fields_future_shape[:2],np.prod(fields_future_shape[2:])])        
        
        predictions        =self.model((fields_past,fields_future), training=False)
        
        predictions_shape  =predictions.shape
        predictions        =np.reshape(predictions  ,[*predictions_shape  [:2],*fields_past_shape         [2:] ])
        
        figure         =self.plotter('monitor',[label,predictions],[])
        image          =self.converter(figure)        
        with self.writer.as_default():
         tf.summary.image("Predicted Frames", image, step=epoch)

class callback_custom_history(tf.keras.callbacks.Callback):
    
    def __init__(self, plotter, whdf):
        super().__init__()
        self.plotter=plotter
        self.whdf   =whdf
    
    def on_train_begin(self, logs=None):
        self.metric_one  =[]

    def on_train_batch_end(self, batch, logs=None):
        keys=list(logs.keys())
        self.metric_one.append  (logs[keys[0]])
    
    def on_train_end(self, logs=None):
        keys        =list(logs.keys())        
        savefilename=os.path.join(self.model.paths[6],'train','metrics')
        self.whdf(savefilename+'.h5'  ,
                  [keys[0]           ],
                  [self.metric_one   ] )
        fig=self.plotter('metrics'              ,
                         [[keys[0]           ],
                          [self.metric_one   ] ],
                         [['b'               ] ] )
        plt.savefig(savefilename+'.png',format='png')


