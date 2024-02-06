# Physics-Informed Surrogates for EM Dynamics using Transformers and GNNs

This is the official repository for the following work published in [_IET Microwaves, Antennas & Propagation_](https://ietresearch.onlinelibrary.wiley.com/journal/17518733):
* [2024 MAP]() : Noakoasteen. Christodoulou. Peng. Goudos. _**"Physics-Informed Surrogates for Electromagnetic Dynamics using Transformers and GNNs"**_  

This work, in earlier stages of its development, has also been presented at the following conferences:
* [2022 APS/URSI](https://2022apsursi.org/view_paper.php?PaperNum=1388) Noakoasteen. _Time-Domain Electromagnetic Analysis using Graph Neural Networks_  
* [2021 APS/URSI](https://www.2021apsursi.org/view_paper.php?PaperNum=2865) Noakoasteen. _Deep Surrogate Models for Time-Domain Electromagnetic Analysis using Attention_  

This work is an extension of the following publication:
* [2020 OJAP](https://ieeexplore.ieee.org/document/9158400) Noakoasteen. Wang. Peng. Christodoulou. _**"Physics-Informed Deep Neural Networks for Transient Electromagnetic Analysis"**_ [[view repository on GitLab]](https://gitlab.com/oameed/unm_cem_dlfdtd)  

## Acknowledgements

* We would like to thank the University of New Mexico Center for Advanced Research Computing [(CARC)](http://carc.unm.edu/), supported in part by the National Science Foundation, for providing the high performance computing resources used in this work.

* The Transformer implementation used in this project is an adaptation from the Transformer architecture developed in the author's other project [_Deep Learning with TensorFlow 2_](https://gitlab.com/oameed/ml_deeplearning_tf2) which itself is an adaptation from **_TensorFlow's tutorial_** on Transformers: [_Neural machine translation with a Transformer and Keras_](https://www.tensorflow.org/text/tutorials/transformer).

* The Graph Neural Network (GNN) implementation used in this project is an adaptation from the GNN Node Classifier developed in the author's other project [_Graph Learning with TensorFlow 2_](https://gitlab.com/oameed/ml_graphlearning_tensorflow) which itself is an adaptation from **_Keras Code Examples_**: [_Node Classification with Graph Neural Networks_](https://keras.io/examples/graph/gnn_citations/). 

## References

**_Similar Works_**  

[[ 1 ]](https://arxiv.org/abs/2212.12794). 2022. Lam. _GraphCast: Learning skillful medium-range global weather forecasting_  
[[ 2 ]](https://www.sciencedirect.com/science/article/pii/S0893608021004500). 2022. Geneva. _Transformers for modeling physical systems_  
[[ 3 ]](https://arxiv.org/abs/2201.09113). 2022. Han. _Predicting Physics in Mesh-reduced Space with Temporal Attention_  
[[ 4 ]](https://arxiv.org/abs/2010.03409). 2020. Pfaff. _Learning Mesh-Based Simulation with Graph Networks_ [[GitHub]](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets) [[blog]](https://sites.google.com/view/meshgraphnets)  
[[ 5 ]](https://arxiv.org/abs/2002.09405). 2020. Sanchez-Gonzalez. _Learning to Simulate Complex Physics with Graph Networks_ [[GitHub]](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate) [[blog]](https://sites.google.com/view/learning-to-simulate)  
[[ 6 ]](https://arxiv.org/abs/2001.08317). 2020. Wu. Green. Ben. _Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case_  

_Useful Web Tutorials_

[[ 7 ]](https://towardsdatascience.com/how-to-use-transformer-networks-to-build-a-forecasting-model-297f9270e630). 2021. Mansar. How to use Transformer Networks to build a Forecasting model  
[[ 8 ]](https://medium.com/mlearning-ai/transformer-implementation-for-time-series-forecasting-a9db2db5c820). 2021. Klingenbrunn. Transformers for Time-series Forecasting  
[[ 9 ]](https://towardsdatascience.com/what-is-teacher-forcing-3da6217fed1c). 2019. Wong. What is Teacher Forcing?  
[[10]](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04). 2019. Maxime. What is a Transformer?  
[[11]](https://jalammar.github.io/illustrated-transformer/). 2018. Alammar. The Illustrated Transformer  
[[12]](http://nlp.seas.harvard.edu/2018/04/03/attention.html). 2018. harvardnlp. The Annotated Transformer  
[[13]](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/). 2017. Brownlee. What is Teacher Forcing for Recurrent Neural Networks?  
[[14]](https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/). 2017. Brownlee. How to Get Started with Deep Learning for Natural Language Processing  
[[15]](https://machinelearningmastery.com/what-are-word-embeddings/). 2017. Brownlee. What Are Word Embeddings for Text?  

## How to Run

* The Transformer is implemented using `Tensorflow` Version `2.11` along with `cudatoolkit` Version `11.8.0` and `cudnn` Version `8.4.1.50`.  
* The GNN is implemented using `Tensorflow` Version `2.2` along with `cudatoolkit` Version `10.1.243` and `cudnn` Version `7.6.5`.   
* The `YAML` files for creating the conda environments used to run this project are included in `run/conda`.    

* To run experiments, with experiment `v12` (using dataset `type2`) as an example:
  1. On the Local Machine, Generate the Dataset:
     * `cd` to Main Project Directory
     * `.\run\ps1\generate_datasets_type_2.ps1`
  2. On the HPC System   , Train the Network and Generate Predictions:
     * `cd` to `run/slrm`
     * `sbatch train_v12.sh`

## Experiments v1x: _The Transformer (Continuous Predictions Mode)_  

**_Type-1: Samples of Continuous Predictions using 128x128 Testing Examples_**

|     |     |     |
|:---:|:---:|:---:|
![][fig_v1_1_7]  | ![][fig_v1_1_8]  | ![][fig_v1_1_9]
![][fig_v1_1_10] | ![][fig_v1_1_11] | ![][fig_v1_1_12]

[fig_v1_1_7 ]:networks/v11/predictions/simulation_13.gif  
[fig_v1_1_8 ]:networks/v11/predictions/simulation_43.gif  
[fig_v1_1_9 ]:networks/v11/predictions/simulation_31.gif  
[fig_v1_1_10]:networks/v11/predictions/simulation_51.gif  
[fig_v1_1_11]:networks/v11/predictions/simulation_20.gif  
[fig_v1_1_12]:networks/v11/predictions/simulation_47.gif  

**_Type-2: Samples of Continuous Predictions using 128x128 Testing Examples_**

|     |     |     |
|:---:|:---:|:---:|
![][fig_v1_2_7]  | ![][fig_v1_2_8]  | ![][fig_v1_2_9]
![][fig_v1_2_10] | ![][fig_v1_2_11] | ![][fig_v1_2_12]

[fig_v1_2_7 ]:networks/v12/predictions/simulation_98.gif  
[fig_v1_2_8 ]:networks/v12/predictions/simulation_100.gif  
[fig_v1_2_9 ]:networks/v12/predictions/simulation_56.gif  
[fig_v1_2_10]:networks/v12/predictions/simulation_47.gif  
[fig_v1_2_11]:networks/v12/predictions/simulation_83.gif  
[fig_v1_2_12]:networks/v12/predictions/simulation_74.gif  

**_Type-3: Samples of Continuous Predictions using 128x128 Testing Examples_**

|     |     |     |
|:---:|:---:|:---:|
![][fig_v1_3_7]  | ![][fig_v1_3_8]  | ![][fig_v1_3_9]
![][fig_v1_3_10] | ![][fig_v1_3_11] | ![][fig_v1_3_12]

[fig_v1_3_7 ]:networks/v13/predictions/simulation_3.gif  
[fig_v1_3_8 ]:networks/v13/predictions/simulation_8.gif  
[fig_v1_3_9 ]:networks/v13/predictions/simulation_34.gif  
[fig_v1_3_10]:networks/v13/predictions/simulation_36.gif  
[fig_v1_3_11]:networks/v13/predictions/simulation_38.gif  
[fig_v1_3_12]:networks/v13/predictions/simulation_39.gif  

## Experiments v2x: _The GNN (Testing Predictions Mode)_   

**_Type-1: Samples of Testing Predictions using 128x128 Testing Examples_**

|     |     |     |
|:---:|:---:|:---:|
![][fig_v2_1_7]  | ![][fig_v2_1_8]  | ![][fig_v2_1_9]
![][fig_v2_1_10] | ![][fig_v2_1_11] | ![][fig_v2_1_12]

[fig_v2_1_7 ]:networks/v21/predictions/simulation_13.gif  
[fig_v2_1_8 ]:networks/v21/predictions/simulation_43.gif  
[fig_v2_1_9 ]:networks/v21/predictions/simulation_31.gif  
[fig_v2_1_10]:networks/v21/predictions/simulation_51.gif  
[fig_v2_1_11]:networks/v21/predictions/simulation_20.gif  
[fig_v2_1_12]:networks/v21/predictions/simulation_47.gif  

**_Type-2: Samples of Testing Predictions using 128x128 Testing Examples_**

|     |     |     |
|:---:|:---:|:---:|
![][fig_v2_2_7]  | ![][fig_v2_2_8]  | ![][fig_v2_2_9]
![][fig_v2_2_10] | ![][fig_v2_2_11] | ![][fig_v2_2_12]

[fig_v2_2_7 ]:networks/v22/predictions/simulation_98.gif  
[fig_v2_2_8 ]:networks/v22/predictions/simulation_100.gif  
[fig_v2_2_9 ]:networks/v22/predictions/simulation_56.gif  
[fig_v2_2_10]:networks/v22/predictions/simulation_47.gif  
[fig_v2_2_11]:networks/v22/predictions/simulation_83.gif  
[fig_v2_2_12]:networks/v22/predictions/simulation_74.gif  

**_Type-3: Samples of Testing Predictions using 128x128 Testing Examples_**

|     |     |     |
|:---:|:---:|:---:|
![][fig_v2_3_7]  | ![][fig_v2_3_8]  | ![][fig_v2_3_9]
![][fig_v2_3_10] | ![][fig_v2_3_11] | ![][fig_v2_3_12]

[fig_v2_3_7 ]:networks/v23/predictions/simulation_3.gif  
[fig_v2_3_8 ]:networks/v23/predictions/simulation_8.gif  
[fig_v2_3_9 ]:networks/v23/predictions/simulation_34.gif  
[fig_v2_3_10]:networks/v23/predictions/simulation_36.gif  
[fig_v2_3_11]:networks/v23/predictions/simulation_38.gif  
[fig_v2_3_12]:networks/v23/predictions/simulation_39.gif  

## Dataset

|     |**Size**|**Fields**|**Objects**|
|:---:|:---:   |:---      |:---       |
**_Type1_** | 128x128  | Total Field / Scattered Field (TF/SF) Excitation   | PEC Objects (Circular, Rectangular or Both) at Random Locations |
**_Type2_** | 128x128  | Single Point Source Excitation at Random Locations | PEC Objects (Circular, Rectangular or Both) at Random Locations |
**_Type3_** | 128x128  | Two Point Source Excitations at Random Locations   | Single Circular PEC Object at a Single Location                 |

### Samples of 128x128 Training Examples

**_Type1_**  

|     |     |     |
|:---:|:---:|:---:|
![][fig_v1_1_1] | ![][fig_v1_1_2] | ![][fig_v1_1_3]
![][fig_v1_1_4] | ![][fig_v1_1_5] | ![][fig_v1_1_6]

[fig_v1_1_1]:data/type1/gif/raw/simulation_24.gif
[fig_v1_1_2]:data/type1/gif/raw/simulation_30.gif
[fig_v1_1_3]:data/type1/gif/raw/simulation_40.gif
[fig_v1_1_4]:data/type1/gif/raw/simulation_46.gif
[fig_v1_1_5]:data/type1/gif/raw/simulation_23.gif
[fig_v1_1_6]:data/type1/gif/raw/simulation_22.gif

**_Type2_**  

|     |     |     |
|:---:|:---:|:---:|
![][fig_v1_2_1] | ![][fig_v1_2_2] | ![][fig_v1_2_3]
![][fig_v1_2_4] | ![][fig_v1_2_5] | ![][fig_v1_2_6]

[fig_v1_2_1]:data/type2/gif/raw/simulation_40.gif
[fig_v1_2_2]:data/type2/gif/raw/simulation_44.gif
[fig_v1_2_3]:data/type2/gif/raw/simulation_38.gif
[fig_v1_2_4]:data/type2/gif/raw/simulation_41.gif
[fig_v1_2_5]:data/type2/gif/raw/simulation_95.gif
[fig_v1_2_6]:data/type2/gif/raw/simulation_77.gif

**_Type3_**  

|     |     |     |
|:---:|:---:|:---:|
![][fig_v1_3_1] | ![][fig_v1_3_2] | ![][fig_v1_3_3]
![][fig_v1_3_4] | ![][fig_v1_3_5] | ![][fig_v1_3_6]

[fig_v1_3_1]:data/type3/gif/raw/simulation_2.gif
[fig_v1_3_2]:data/type3/gif/raw/simulation_4.gif
[fig_v1_3_3]:data/type3/gif/raw/simulation_5.gif
[fig_v1_3_4]:data/type3/gif/raw/simulation_6.gif
[fig_v1_3_5]:data/type3/gif/raw/simulation_43.gif
[fig_v1_3_6]:data/type3/gif/raw/simulation_47.gif

## Code Statistics

<pre>
github.com/AlDanial/cloc v 1.96  T=0.33 s (90.2 files/s, 9901.5 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                          10            178            196           1187
MATLAB                           7             88            160            959
Markdown                         1             52              0            163
Bourne Shell                     6             44             42             90
PowerShell                       6             44             12             78
-------------------------------------------------------------------------------
SUM:                            30            406            410           2477
-------------------------------------------------------------------------------
</pre>

