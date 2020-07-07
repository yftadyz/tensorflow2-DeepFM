# ---- build a binary classification model on titanic dataset ----
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#from importlib import reload
#import model
#reload(model)
from model import deepFM
from utils import fill_na,preprocess

# --------- prepare dataset -------

#read data
dftrain = pd.read_csv('titanic-train.csv')
dfeval = pd.read_csv('titanic-eval.csv')
dftrain.info()

#fill na values
fill_na(dftrain)
fill_na(dfeval)

#preprocess dataset
meta_cate=preprocess(dftrain,
           cate_cols=['sex','class','deck','embark_town','alone']
            )
preprocess(dfeval,
           cate_cols=['sex','class','deck','embark_town','alone'],
           existed_cate=meta_cate)

#target column
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#transform dataframe to tensor
trainset=tf.data.Dataset.from_tensor_slices((dftrain.to_dict('list'),y_train.values))
evalset=tf.data.Dataset.from_tensor_slices((dfeval.to_dict('list'),y_eval.values))

#prepare batch set
batch_size=32
epoch_num=20
epoch_size=dftrain.shape[0]//batch_size
if dftrain.shape[0]%batch_size!=0:
    epoch_size+=1
trainset=trainset.shuffle(dftrain.shape[0]).batch(batch_size).repeat(epoch_num)
evalset=evalset.shuffle(dfeval.shape[0]).batch(128).repeat(1)

# ----- train ------
i=0
ob_steps=5
loss_sum=0
lamda=0.0005

#set evaluation metric
mtc=tf.metrics.BinaryAccuracy()
mtc2=tf.metrics.BinaryAccuracy()

#initalize model
parameters={}
parameters['fm_cols']=['sex', 'age', 'n_siblings_spouses', 'parch', 'fare',
       'class', 'deck', 'embark_town', 'alone']
parameters['fm_emb_dim']=32
parameters['hidden_units']=[32,16]
parameters['dropprob']=0.3
mymodel=deepFM(parameters)

#set loss func
loss=tf.losses.BinaryCrossentropy()

#set optimizer
opt=tf.optimizers.Adam(learning_rate=0.01)

#record logloss and acc of every ob_step steps
logloss=[]
logloss2=[]
acc=[]
acc2=[]

for ft,label in trainset:
    print("Step %d ...."%i)
    with tf.GradientTape() as tape:
        cur_loss = loss(label, mymodel(ft,training=True))\
        +lamda/(2*batch_size)*sum([tf.reduce_sum(tf.square(w)) for w in mymodel.trainable_variables])
    #compute gradient
    grad=tape.gradient(cur_loss,mymodel.trainable_variables)
    #update parameters
    opt.apply_gradients(zip(grad,mymodel.trainable_variables))

    loss_sum+=cur_loss
    mtc.update_state(label, mymodel(ft))

    if i%ob_steps==0:
        #get train metrics
        acc.append(mtc.result().numpy())
        logloss.append(loss_sum/ob_steps)
        loss_sum=0
        mtc.reset_states()

        #get eval metrics
        loss_sum2=0
        j=0
        mtc2.reset_states()
        for ft2, label2 in evalset:
            mtc2.update_state(label2, mymodel(ft2))
            loss_sum2+=loss(label2, mymodel(ft2))
            j+=1
        logloss2.append(loss_sum2/j)
        acc2.append(mtc2.result().numpy())
    i+=1

print("Best Accuracy on EVALset: %.4f"%max(acc2))

# --------- Visualization --------
# plot logloss curve
img=plt.figure(figsize=(16,8))

plt.subplot(1,2,1)
plt.title("Titanic: LogLoss")
x=[i*ob_steps/epoch_size for i in range(len(logloss))]
plt.plot(x,logloss)
plt.plot(x,logloss2)
plt.legend(['Train Logloss','Eval Logloss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(epoch_num))
plt.tight_layout()

plt.subplot(1,2,2)
plt.title("Titanic: Accuracy")
x=[i*ob_steps/epoch_size for i in range(len(acc))]
plt.plot(x,acc)
plt.plot(x,acc2)
plt.legend(['Train Acc','Eval Acc'])
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.xticks(range(epoch_num))
plt.tight_layout()

img.show()
img.savefig('result.png')