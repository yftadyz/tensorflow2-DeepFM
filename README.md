# Tensorflow2-DeepFM
This project includes a simple DeepFM[1] implementation with tensorflow2. Implementation is inspired by [2]. 

# Features
Convenient to train this model on a pandas dataframe dataset. 

# Usage
An example of Titanic is included to show how to use it.
```
parameters={}
parameters['fm_cols']=['sex', 'age', 'n_siblings_spouses', 'parch', 'fare',
       'class', 'deck', 'embark_town', 'alone']
parameters['fm_emb_dim']=32
parameters['hidden_units']=[32,16]
parameters['dropprob']=0.3
mymodel=deepFM(parameters)
```
Just set column names you want to use in your dataframe and model struture, like embedding dimension, hidden layer structures and dropout prob.

# Reference
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.

[2] https://github.com/ChenglongChen/tensorflow-DeepFM
