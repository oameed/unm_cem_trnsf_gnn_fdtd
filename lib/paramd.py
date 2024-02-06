##############################################
### UNIVERSITY OF NEW MEXICO               ###
### COMPUTATIONAL EM LAB                   ###
### EMULATING FDTD USING TRANSFORMER & GNN ###
### PARAMETER DEFINITIONS                  ###
### by: OAMEED NOAKOASTEEN                 ###
##############################################

import os
import argparse

### DEFINE PARSER
parser=argparse.ArgumentParser() 

### DEFINE PARAMETERS
parser.add_argument('-data'      , type=str  ,               help='NAME-OF-DATA-TYPE-DIRECTORY'       ,required=True         )
parser.add_argument('-net'       , type=str  , default='v00',help='NAME-OF-NETWORK-PROJECT'                                  )
  # GROUP: HYPER-PARAMETERS FOR PRE-PROCESSING OF DATA
parser.add_argument('-simb'      , type=int  , default=190  ,help='SIMULATION-BEGINNING-FRAME'                               )
parser.add_argument('-sime'      , type=int  , default=400  ,help='SIMULATION-END-FRAME'                                     )
parser.add_argument('-W'         , type=int  , default=128  ,help='WIDTH-OF-IMAGE'                                           )
parser.add_argument('-H'         , type=int  , default=128  ,help='HEIGHT-OF-IMAGE'                                          )
parser.add_argument('-list'      ,                           help='GENERATE-RANDOM-TRAIN-TEST-LISTS'  ,action  ='store_true' )
parser.add_argument('-div'       , type=float, default=0.75 ,help='PORTION-OF-TOTAL-ASSIGNED-TO-TRAIN'                       )
  # GROUP: HYPER-PARAMETERS FOR TRAINING AND TESTING
parser.add_argument('-b'         , type=int  , default=4    ,help='BATCH-SIZE'                                               )
parser.add_argument('-t'         , type=int  , default=5    ,help='VIDEO-SIZE'                                               )
parser.add_argument('-c'         , type=int  , default=3    ,help='NUMBER-OF-CHANNELS'                                       )
parser.add_argument('-eptr'      , type=int  , default=250  ,help='NUMBER-OF-TRAINING-EPOCHS'                                )
parser.add_argument('-steptr'    , type=int  , default=50   ,help='EVERY-TRAIN-STEPS-TO-WRITE-SUMMERIES'                     )
parser.add_argument('-epts'      , type=int  , default=1    ,help='NUMBER-OF-TESTING-EPOCHS'                                 )
parser.add_argument('-stepts'    , type=int  , default=1    ,help='EVERY-TEST-STEPS-TO-WRITE-SUMMERIES'                      )
parser.add_argument('-bc'        , type=int  , default=10000,help='BATCH-BUFFER-CAPACITY'                                    )
parser.add_argument('-lr'        , type=float, default=0.001,help='LEARNING-RATE'                                            )
parser.add_argument('-ckptn'     , type=int  , default=1    ,help='NUMBER-OF-CHECKPOINTS-TO-KEEP'                            )
parser.add_argument('-gdl'       , type=float, default=0.004,help='GDL-LOSS-COEFFICIENT'                                     )
parser.add_argument('-nointscale',                           help='NO INTERNAL SCALING IN NETWORK'    ,action  ='store_true' )
parser.add_argument('-sp'        , type=int  , default=10   ,help='NUMBER OF PAST FRAMES FOR THE TRANSFORMER'                )
parser.add_argument('-ex'        ,                           help='EXIT RUN FOR TIMING STUDIES'       ,action  ='store_true' )

### ENABLE FLAGS
args=parser.parse_args()

### CONSTRUCT PARAMETER STRUCTURES
PATHS    =[os.path.join('..','..','data'    ,args.data,'hdf5'       ,'raw'              ),
           os.path.join('..','..','data'    ,args.data,'hdf5'       ,'processed'        ),           
           os.path.join('..','..','data'    ,args.data,'info'                           ),
           os.path.join('..','..','networks',args.net ,'model'                          ),
           os.path.join('..','..','networks',args.net ,'tfrecords'                      ),
           os.path.join('..','..','networks',args.net ,'checkpoints',                   ),
           os.path.join('..','..','networks',args.net ,'logs'                           ),
           os.path.join('..','..','networks',args.net ,'predictions','hdf'              ) ]           

 
FRAMEMNGM=[args.simb,args.sime ,args.W     ,args.H                            ]
genLIST  = args.list
trainDIV = args.div
NETV     = args.net
numT     = args.t
numC     = args.c
BATCH    =[args.b   ,args.bc                                                  ]
OPTIM    =[args.lr  ,args.eptr ,args.steptr,args.ckptn,args.epts,args.stepts  ]
GDLFACTOR= args.gdl
VIDFRAME =[args.t   ,args.simb ,args.sime                                     ]
INTSCALE = args.nointscale

PARAMS   =[[args.simb, args.sime      , args.W     , args.H                                                        ],
           [args.list, args.div                                                                                    ],
           [args.net , args.t         , args.c     , args.b    , args.bc                                           ],
           [args.lr  , args.eptr      , args.steptr, args.ckptn, args.epts, args.stepts                            ],
           [args.gdl , args.nointscale                                                                             ],
           [3        , 256            , 512        , 8         , 0.1      , 7765       , 7010, 1000, 1000, args.sp ] ]

if   args.data in ['type1']:
 PARAMS.append  (['simulation_20.h5']) 
else:
 if  args.data in ['type2']:
  PARAMS.append (['simulation_98.h5'])
 else:
  if args.data in ['type3']:
   PARAMS.append(['simulation_8.h5' ])

PARAMS.append([[32,32],
                0.5   ,
                0.01   ])

PARAMS.append([args.ex])

# PATHS[0]    :                   HDFRAWPATH   
# PATHS[1]    :                   HDFPRCPATH   
# PATHS[2]    :                   INFOPATH     
# PATHS[3]    :                   MODELPATH    
# PATHS[4]    :                   TFRPATH      
# PATHS[5]    :                   CKPTPATH     
# PATHS[6]    :                   LOGPATH      
# PATHS[7]    :                   VIDPREDPATH  

# FRAMEMNGM[0]: BEGINNING INDEX OF THE EVENTFUL PORTION OF THE SIMULATION 
# FRAMEMNGM[1]: ENDING    INDEX OF THE EVENTFUL PORTION OF THE SIMULATION 
# FRAMEMNGM[2]: WIDTH  OF EACH FRAME
# FRAMEMNGM[3]: HEIGHT OF EACH FRAME
# genLIST     : GENERATE TRAIN/TEST LISTS
# trainDIV    : TRAIN/TEST DIVITION: NUMBER OF TRAIN SAMPLES ARE DIV*TOTAL
# NETV        : NETWORK VERSION
# numT        : NUMBER OF TIME-FRAMES (VIDEO SIZE)
# numC        : NUMBER OF INPUT CHANNELS
# BATCH[0]    : BATCH SIZE
# BATCH[1]    : BATCH CAPACITY
# OPTIM[0]    : LEARNING RATE
# OPTIM[1]    : NUMBER OF EPOCHS FOR TRAINING
# OPTIM[2]    : STEPS: EVERY 'STEPS' WRITE SUMMARY/CHECKPOINT FOR TRAINING
# OPTIM[3]    : MAXIMUM NUMBER OF CHECKPOINTS TO KEEP
# OPTIM[4]    : NUMBER OF EPOCHS FOR TESTING
# OPTIM[5]    : STEPS: EVERY 'STEPS' WRITE SUMMARY/CHECKPOINT FOR TESTING
# GDLFACTOR   : GDL LOSS COEFFICIENT
# VIDFRAME    : VIDEO FRAME MANAGEMENT FOR VIDEO GENERATION

# PARAMS[0][0]: FRAME MANAGEMENT: BEGGINING FRAME
# PARAMS[0][1]: FRAME MANAGEMENT: END       FRAME
# PARAMS[0][2]: FRAME MANAGEMENT: WIDTH  OF FRAME
# PARAMS[0][3]: FRAME MANAGEMENT: HEIGHT OF FRAME

# PARAMS[1][0]: PREPROCESSING   : GENERATE TRAIN/TEST LIST: True/False
# PARAMS[1][1]: PREPROCESSING   : TRAIN TO TEST RATIO

# PARAMS[2][0]: TRAINING PARAMS : NETV 
# PARAMS[2][1]: TRAINING PARAMS : numT
# PARAMS[2][2]: TRAINING PARAMS : numC
# PARAMS[2][3]: TRAINING PARAMS : BATCH SIZE
# PARAMS[2][4]: TRAINING PARAMS : BATCH CAPACITY

# PARAMS[3][0]: TRAINING PARAMS : LEARNING RATE
# PARAMS[3][1]: TRAINING PARAMS : NUMBER OF TRAIN EPOCHS
# PARAMS[3][2]: TRAINING PARAMS : EVERY STEPS TAKE ACTION DURING TRAINING
# PARAMS[3][3]: TRAINING PARAMS : NUMBER OF CHECKPOINTS TO KEEP
# PARAMS[3][4]: TRAINING PARAMS : NUMBER OF TEST  EPOCHS
# PARAMS[3][5]: TRAINING PARAMS : EVERY STEPS TAKE ACTION DURING TESTING

# PARAMS[4][0]: OTHER           : GDL LOSS FACTOR
# PARAMS[4][1]: OTHER           : NO INTERNAL SCALING IN NETWORK

# PARAMS[5][0]: TRANSFORMER     : NUMBER OF LAYERS
# PARAMS[5][1]: TRANSFORMER     : DEPTH  OF THE MODEL
# PARAMS[5][2]: TRANSFORMER     : DEPTH  OF THE FEED FORWARD NETWORK 
# PARAMS[5][3]: TRANSFORMER     : NUMBER OF HEADS
# PARAMS[5][4]: TRANSFORMER     : DROPOUT RATE
# PARAMS[5][5]: TRANSFORMER     : INPUT  (PT) VOCAB SIZE
# PARAMS[5][6]: TRANSFORMER     : TARGET (EN) VOCAB SIZE
# PARAMS[5][7]: TRANSFORMER     : INPUT  MAXIMUM POSITIONAL ENCODING
# PARAMS[5][8]: TRANSFORMER     : OUTPUT MAXIMUM POSITIONAL ENCODING
# PARAMS[5][9]: TRANSFORMER     : NUMBER OF FRAMES AS CONTEXT/PAST


# PARAMS[6][0]: TRANSFORMER/GNN : DATASET SIMULATION NAME FOR MONITORING DATA 

# PARAMS[7][0]: GNN             : HIDDEN UNITS
# PARAMS[7][1]: GNN             : DROPOUT RATE
# PARAMS[7][2]: GNN             : LEARNING RATE

