##Running for Python 3.6
##Generating keras modeling to predict sleep-wake classification

import tensorflow as tf
from tensorflow import keras
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
from numpy import genfromtxt
from datetime import datetime
import random
from sklearn.model_selection import train_test_split ##Need to randomly split data
from main import CONFIG_PATH

from hypnogram_postprocess import postProcessHypnogram
from helper_functions import *
from sklearn.preprocessing import normalize
from operator import itemgetter


#from keras.layers import Embedding,  LSTM
from main import CONFIG_PATH
config = load_config(CONFIG_PATH / "my_config.yaml")
globals().update(config)

def initializeModel():

    
    ## Batch normalazition 
    model = keras.Sequential()

    model.add(keras.layers.Dense(10,activation='relu', input_shape=(319,)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Dropout(0.5))
	
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.BatchNormalization())
	
    model.add(keras.layers.Dense(12, activation='relu'))
    model.add(keras.layers.BatchNormalization())
	
    model.add(keras.layers.Dense(4, activation='softmax',input_shape=(4,)))
    
    return model


## Assumes that the input file has the same name as the data, but with the word '_labels' 
## appended to the end of the file. Assumes labels are already one-hot encoded
def loadInputData(filename):
    DATA_DIR = WORKING_DIR + filename
    LABEL_DIR = DATA_DIR.replace('_data.csv', '_labels.csv')
    print('Loaded labels for: ' + filename)
    print(DATA_DIR)
    
    ## Now, call pandas and load data
    train_data = pd.read_csv(DATA_DIR, header=None, delimiter=',').to_numpy()
    train_labels = pd.read_csv(LABEL_DIR, header=None, delimiter=',').to_numpy()
	
    return train_data, train_labels

## Techniques to rebalance the data would be either downsaampling the majority class or artificially 
## resampling the minority class using SMOTE. Current method is simply downsampling
def rebalanceData(x_train, y_train):
    ##Check class distribution
    storeAllocated = []
    x_out = np.empty((0,319))
    y_out = np.empty((0,4))
    [print("s" + str(s) + ":" +  str(len(np.argwhere(y_train[:,s] == 1)))) for s in range(0,4)]

    for s in range(0,4):
        toExtract = np.argwhere(y_train[:,s] == 1).flatten()
        storeAllocated.append(toExtract)#y_train[toExtract,])
    minNum = len(storeAllocated[3])
    
    ##For Awake, NREM, and REM. For unclassified, append as is, or remove half
    for s in [0,2,3]:
        toAllocate = random.sample([storeAllocated[s][i] for i in range(0,len(storeAllocated[s]))], minNum)
        x_out = np.append(x_out, np.array(x_train[toAllocate,:]), axis=0)
        y_out = np.append(y_out, np.array(y_train[toAllocate,:]), axis=0)
        
    x_out = np.append(x_out, np.array(x_train[storeAllocated[1],:]), axis=0)
    y_out = np.append(y_out, np.array(y_train[storeAllocated[1],:]), axis=0)
    
    return x_out, y_out
	
## Feed in data to the neural network that has already been windowed
## Output is samples x freq x windows. This way, the shape is correct while preserving memory
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def my_learning_rate(epoch):
    print(epoch)
    return 1e-2/(epoch/2+1)

def run():
    model = initializeModel()    
    model.compile(optimizer='adam',#tf.keras.optimizers.Adam(lr=1e-2),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])
    lrs = keras.callbacks.LearningRateScheduler(my_learning_rate)
    print(model.summary())
    print('This is WORKING_DIR')
    print(WORKING_DIR)
    
    filename = '/model_spectrogram_' + SESSION_ID + '_data.csv'
    train_data, train_labels = loadInputData(filename)
    
    hypnogramOrig = onehot_to_array(train_labels)
    hypTimeAxis = np.linspace(1/hypFs, len(hypnogramOrig)/hypFs, len(hypnogramOrig))
    print('Here are the shapes of our dataset')
    print('Data: ' + str(train_data.shape))
    print('Labels: ' + str(train_labels.shape))
    print(train_data[:10,:])
    print(train_labels[:10,:])
    
    acc, val_acc, loss,val_loss = [],[],[],[]
    

    ## Running k-folds classification to improve generalization and reduce overfitting
    for i in range(0,10):
    
        ##Only need to balance the training data, not the validation data.
        x_train, x_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=0.2, shuffle= True)
        x_train, y_train = rebalanceData(x_train, y_train) ##Utilize a form of downsampling rather than synthetic data
        history = model.fit(x_train, y_train, batch_size=900, epochs=100, validation_data=(x_valid, y_valid), verbose=1)#, callbacks=[lrs])
		
        # acc.append(history.history['accuracy'])
        # val_acc.acc(history.history['val_accuracy'])
        acc.append(history.history['acc'])
        val_acc.append(history.history['val_acc'])
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
    
    print("Finished model fitting")
    
    
    print("Starting prediction...")
    predictedHypnogram = model.predict(train_data)
    # Serialize model to JSON
    predictedHypnogram = np.array(predictedHypnogram)
    
    today = datetime.now().strftime('%m_%d_%Y_%H-%M')
    saveName = WORKING_DIR+nameappend+ "_date-" + today+ "_model"
    print('Saving the model on ' + today + " to " + saveName)
    model.save(saveName + "_iter0.h5")
    print("Saved model to disk")
    
    print("Starting post-processing...this may take a while")
    newPredict, newProbVal = postProcessHypnogram(predictedHypnogram)
    epochs = range(len(acc))
	
	# Plot evaluations
    plt.figure()
    plt.subplot(211)
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.subplot(212)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.show()
    
	# Plot predicted hypnogram against the ground truth
    storeAcc, hFig = plotValidation(hypTimeAxis, hypnogramOrig, newPredict) ## Plot both the original and DNN output
