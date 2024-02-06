###############################################
### UNIVERSITY OF NEW MEXICO                ###
### COMPUTATIONAL EM LAB                    ###
### EMULATING FDTD USING TRANSFORMER & GNN  ###
### INPUT PIPELINE FUNCTION DEFINITIONS     ###
### by: OAMEED NOAKOASTEEN                  ###
###############################################

import os
import numpy      as  np
import tensorflow as  tf
from utilsd import   (getFILENAMES,
                      rHDF        ,
                      rFILE        )

def get_data(PARAMS,FILENAME):
 def getCHANNELS(DATA,CN):
  # THIS FUNCTION RETURNS APPROPRIATE FIELD COMPONENTS FOR THE GIVEN NUMBER OF CHANNELS
  # THE ORDER OF FIELD DATA IN CHANNELS IS Ex,Ey,Hz
  # FOR CN==1 ONLY Hz IS RETURNED
  # FOR CN==2 Ex AND Ey ARE RETURNED
  if CN==1:
   data=DATA[:,:,:,2]
   data=np.expand_dims(data,axis=-1)
  else:
   if CN==2:
    data=DATA[:,:,:,0:2]
   else:
    if CN==3:
     data=DATA
  return data
 SHAPE       =[PARAMS[2][1],PARAMS[2][2]]
 fobj,keys   =rHDF(FILENAME)  
 img         =getCHANNELS(fobj['img'],SHAPE[1])                                                        # LOAD 'numC' CHANNELS FROM FIELD DATA
 bnd         =            fobj['bnd']
 shape       =img.shape
 img         =np.reshape(img,[int(np.floor(shape[0]/SHAPE[0])),SHAPE[0],shape[1],shape[2],shape[3]])   # RESHAPE TO VIDEO FORMAT
 bnd         =np.reshape(bnd,[int(np.floor(shape[0]/SHAPE[0])),SHAPE[0],shape[1],shape[2],1       ]) 
 return img, bnd

def get_graph_info(PATHS,PARAMS):
 def create_graph(PARAMS):  
  shape         =[PARAMS[0][2],PARAMS[0][3],PARAMS[2][2]]
  list_of_tuples=[]
  edges         =[]
  for  i in range(shape[0]):
   for j in range(shape[1]):
    list_of_tuples.append((i,j))
  for  i in range(len(list_of_tuples)):
   neighbor_up   =(list_of_tuples[i][0]+1,list_of_tuples[i][1]  )
   try:
    index        =list_of_tuples.index(neighbor_up   )
    edges.append((i,index))
   except ValueError:
    pass
   neighbor_down =(list_of_tuples[i][0]-1,list_of_tuples[i][1]  )
   try:
    index        =list_of_tuples.index(neighbor_down )
    edges.append((i,index))
   except ValueError:
    pass
   neighbor_left =(list_of_tuples[i][0]  ,list_of_tuples[i][1]-1)
   try:
    index        =list_of_tuples.index(neighbor_left )
    edges.append((i,index))
   except ValueError:
    pass
   neighbor_right=(list_of_tuples[i][0]  ,list_of_tuples[i][1]+1)
   try:
    index        =list_of_tuples.index(neighbor_right)
    edges.append((i,index))
   except ValueError:
    pass
  return np.array(list_of_tuples), np.array(edges)
 savefilenames =[os.path.join(PATHS[2],'graph_vortices'+'.csv'),
                 os.path.join(PATHS[2],'graph_edges'   +'.csv') ]
 filenames     =getFILENAMES (PATHS[2])
 if not all(x in filenames for x in savefilenames):
  list_of_tuples,edges=create_graph(PARAMS)
  np.savetxt(savefilenames[0], list_of_tuples, delimiter=",")
  np.savetxt(savefilenames[1], edges         , delimiter=",")
 else:
  list_of_tuples      =np.loadtxt(savefilenames[0], delimiter=",")
  edges               =np.loadtxt(savefilenames[1], delimiter=",")
 return edges

def saveTFRECORDS(PATHS, PARAMS, LIST):
 def get_filename(FN,PATHS):
  filename=FN.split('.')[0]+'.tfrecords'
  filepath=os.path.join(PATHS[4],'train' )
  return os.path.join  (filepath,filename)
 def serialize_example(IMG,BND):
  feature      ={'img'    : tf.train.Feature(bytes_list=tf.train.BytesList(value=[IMG.tostring()])), 
                 'bnd'    : tf.train.Feature(bytes_list=tf.train.BytesList(value=[BND.tostring()])) }
  example_proto=tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()
 def write_serialized_example(IMG,BND,FILENAME):
  with tf.io.TFRecordWriter(FILENAME) as writer:
   for i in range(IMG.shape[0]):
    example=serialize_example(IMG[i],BND[i])
    writer.write(example)
 filenames=[(fn,os.path.join(PATHS[1],fn)) for fn in rFILE(LIST)]
 for fn, full_fn in filenames:
  img,bnd     =get_data    (PARAMS,full_fn              ) 
  savefilename=get_filename(fn    ,PATHS                )
  write_serialized_example (img   ,bnd    , savefilename)
 
def inputTFRECORDS(PATH, PARAMS):
 SHAPE_IMG =[PARAMS[2][1],PARAMS[0][2],PARAMS[0][3],PARAMS[2][2]]
 SHAPE_BND =[PARAMS[2][1],PARAMS[0][2],PARAMS[0][3],1           ]
 filenames =getFILENAMES(PATH)
 feature   ={'img'    : tf.io.FixedLenFeature([],tf.string), 
             'bnd'    : tf.io.FixedLenFeature([],tf.string) }
 def parse_function(example_proto):
  parsed_example=tf.io.parse_single_example(example_proto,feature )
  _img          =tf.io.decode_raw(parsed_example['img'],tf.float32)
  _bnd          =tf.io.decode_raw(parsed_example['bnd'],tf.float32)
  _img.set_shape([np.prod(SHAPE_IMG)])
  _bnd.set_shape([np.prod(SHAPE_BND)])
  img           =tf.reshape(_img,SHAPE_IMG)
  bnd           =tf.reshape(_bnd,SHAPE_BND)
  return img,bnd
 dataset =tf.data.TFRecordDataset(filenames                                    )
 dataset =dataset.map            (parse_function                               )
 dataset =dataset.batch          (PARAMS[2][3]  , drop_remainder          =True)
 dataset =dataset.shuffle        (PARAMS[2][4]  , reshuffle_each_iteration=True)
 return dataset

