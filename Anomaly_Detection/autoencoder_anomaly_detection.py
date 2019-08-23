# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:37:06 2018
@author: Yesser H. Nasser
"""
'''build an LSTM autoencoder'''
'''
A mutivaraiante time-series data contains multiple varaiables observed 
over a period of time. we will build an LSTM autoencoder to perform
rare-event classification. 
'''
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set(color_codes=True)
import pandas as pd
import numpy as np
import keras

from pylab import rcParams

import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(11)

# help randomly select the data points
SEED = 123
DATA_SPLIT_PCT = 0.2
rcParams['figure.figsize']=8,6
LABELS = ['Normal', 'Break']

data_dir = 'test_data' 
merged_data = pd.DataFrame()
for filename in os.listdir(data_dir):
    print(filename)
    dataset = pd.read_csv(os.path.join(data_dir,filename), sep='\t')
    dataset_mean_abs = np.array(dataset.abs().mean())
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,4))
    dataset_mean_abs.index = [filename]
    merged_data = merged_data.append(dataset_mean_abs)
merged_data.columns = ['Bearing 1','Bearing 2','Bearing 3','Bearing 4']

merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')

merged_data = merged_data.sort_index()
merged_data.to_csv('merged_dataset_BearingTest_2.csv')
print(merged_data.head())
merged_data.plot()

# define train and testing data set
dataset_train = merged_data['2004-02-12 10:32:39':'2004-02-13 23:52:39']
dataset_test = merged_data['2004-02-13 23:52:39':]
plt.figure(figsize=(10,7))
dataset_train.plot()

scaler = preprocessing.MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(dataset_train), columns=dataset_train.columns, index = dataset_train.index)
X_test = pd.DataFrame(scaler.fit_transform(dataset_test), columns=dataset.columns, index=dataset_test.index)


training_epochs = 100
batch_size = 10
input_dim = X_train.shape[1] # number of columns
encoding_dim = 32
hidden_dim = int(encoding_dim/2)
learning_rate = 1e-7

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation='tanh', activity_regularizer=regularizers.l1(learning_rate))(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(hidden_dim, activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

autoencoder.compile(loss = keras.losses.mse, optimizer= keras.optimizers.Adam(), metrics=['accuracy'])
autoencoder_model = autoencoder.fit(np.array(X_train), np.array(X_train), batch_size=batch_size, epochs=training_epochs, validation_split=0.05)

plt.figure(figsize=(8,4))
plt.plot(autoencoder_model.history['loss'], linewidth=2, label='Train')
plt.plot(autoencoder_model.history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylim([0,.02])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

X_pred = autoencoder.predict(np.array(X_train))
X_pred = pd.DataFrame(X_pred, columns=X_train.columns)
X_pred.index = X_train.index
scored = pd.DataFrame(index=X_train.index)
scored['Loss_mse'] = np.mean(np.abs(X_pred - X_train), axis=1)

plt.figure()
sns.distplot(scored['Loss_mse'], bins=10, kde=True, color='blue')
plt.xlim([0,0.05])

X_pred = autoencoder.predict(np.array(X_test))
X_pred = pd.DataFrame(X_pred, 
                      columns=X_test.columns)
X_pred.index = X_test.index

scored = pd.DataFrame(index=X_test.index)
scored['Loss_mse'] = np.mean(np.abs(X_pred-X_test), axis = 1)
scored['Threshold'] = 0.02
scored['Anomaly'] = scored['Loss_mse'] > scored['Threshold']
scored.head()

X_pred_train = autoencoder.predict(np.array(X_train))
X_pred_train = pd.DataFrame(X_pred_train, 
                      columns=X_train.columns)
X_pred_train.index = X_train.index

scored_train = pd.DataFrame(index=X_train.index)
scored_train['Loss_mse'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
scored_train['Threshold'] = 0.02
scored_train['Anomaly'] = scored_train['Loss_mse'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])

scored.plot(logy=True, figsize=(10,16), color=['blue','red'])








