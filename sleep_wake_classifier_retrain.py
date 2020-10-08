# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 08:51:07 2020

@author: danny
"""

import tensorflow as tf
from tensorflow import keras
import scipy.io
#import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import re
from numpy import genfromtxt
from datetime import datetime
import random
#from keras.models import model_from_json
from sklearn.model_selection import train_test_split ##Need to randomly split data
from hypnogram_postprocess import postProcessHypnogram
from helper_functions import *
from sklearn.preprocessing import normalize
from operator import itemgetter
from main import CONFIG_PATH
from sleep_wake_classifier_simpler import *

from main import CONFIG_PATH
config = load_config(CONFIG_PATH / "my_config.yaml")
globals().update(config)

DATA_DIR = '/model_spectrogram_' + SESSION_ID +'_data.csv'
#from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical
#from keras.layers import Embedding,  LSTM

## Finds the iteration number. If it doesn't exist, just append to the end of the name
def getNewIter(saveName):
    iterNum = re.search("iter\d{1,2}", saveName)
    iterNum = "iter0" if iterNum is None else iterNum.group()
    
    ##Temporarily replace to increment string..
    if(str.find(saveName, "iter") == -1):    
        saveName = saveName + "_" + iterNum + ".h5"
    else:
        print(iterNum)
        newNum = "iter" + str(int(re.search("\d{1,2}", iterNum).group())+1)
        saveName = saveName.replace(iterNum, newNum)
    
    return saveName
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
    # train_labels = keras.utils.to_categorical(train_labels) ##For the DNN to recognize
    # train_labels = keras.utils.to_categorical(genfromtxt(LABEL_DIR, delimiter=','))
    return train_data, train_labels

def run():
    # Load new test data
    # SESSION_ID='02-24'
    train_data, train_labels = loadInputData(DATA_DIR)
    # train_data = normalize(train_data, axis=1, norm='l2')
    hypnogramOrig = onehot_to_array(train_labels)
    hypTimeAxis = np.linspace(1/hypFs, len(hypnogramOrig)/hypFs, len(hypnogramOrig))
    # load existing model
    loaded_model = keras.models.load_model(currentModel + ".h5")
    print("Loaded model from disk" + currentModel)
    
    ##Fit new data
    ## Should rename it anyway
    for i in range(0,10):
        
        ##Only need to balance the training data, not the validation data.
        x_train, x_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=0.2, shuffle= True)
        x_train, y_train = rebalanceData(x_train, y_train) ##Utilize a form of downsampling rather than synthetic data
        history = loaded_model.fit(x_train, y_train, batch_size=900, epochs=100, validation_data=(x_valid, y_valid), verbose=1,)
    
    print("Finished model fitting")
    
    
    print("Starting prediction...")
    predictedHypnogram = loaded_model.predict(train_data)
    predictedHypnogram = np.array(predictedHypnogram)

    saveName = getNewIter(currentModel)
    print('Saving the model to: ' + saveName)
    loaded_model.save(saveName +".h5")

    
    print("Starting post-processing...this may take a while")
    newPredict, newProbVal = postProcessHypnogram(predictedHypnogram)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    plt.figure()
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
   
    plt.show()
    
    storeAcc, hFig = plotValidation(hypTimeAxis, hypnogramOrig, newPredict) ## Plot both the original and DNN output
