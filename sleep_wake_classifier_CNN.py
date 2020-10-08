##Running for Python 3.6
##Generating keras modeling to predict sleep-wake classification

#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import scipy.io
#import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
from numpy import genfromtxt
from datetime import datetime
import random
#from keras.models import model_from_json
from sklearn.model_selection import train_test_split ##Need to randomly split data
from hypnogram_postprocess import postProcessHypnogram
from helper_functions import *
from sklearn.preprocessing import normalize,scale
from operator import itemgetter


#from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical
#from keras.layers import Embedding,  LSTM
CONFIG_PATH = '/nfs/turbo/lsa-ojahmed/danny/Sleep-Wake-Classification/BrainState/DNN/python_build/config/'
config = load_config(CONFIG_PATH + "my_config.yaml")
globals().update(config)

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.9):
            print('\Reached 90% accuracy. Stop training')
            self.model.stop_training=True
        

def initializeModel():
    model = keras.Sequential()
    ##model.add(keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(319,)))
    ##model.add(keras.layers.MaxPooling1D(pool_size=2))
    ##model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    ##model.add(keras.layers.MaxPooling2D((2, 2)))
    ##model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    ##model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10,activation='relu', input_shape=(319,)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    ##model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(12, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax',input_shape=(4,)))
    
    return model

def initializeModelConv():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (10,3), activation='elu', input_shape=(319,32,1)))
    model.add(keras.layers.MaxPooling2D((1, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32, (5,3), activation='elu', input_shape=(319,32,1)))
    model.add(keras.layers.MaxPooling2D((1, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32, (3,3), activation='elu', input_shape=(319,32,1)))
    model.add(keras.layers.MaxPooling2D((6, 2)))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(319,32,1)))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    #model.add(keras.layers.MaxPooling1D(pool_size=2))
    ##model.add(keras.layers.Conv2D(64, (3,♣ 3), activation='relu'))
    
    ##model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    #model.add(keras.layers.Dense(64, activation='relu'))
    #model.add(keras.layers.Dense(16, activation='relu'))
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
    # train_labels = keras.utils.to_categorical(train_labels) ##For the DNN to recognize
    # train_labels = keras.utils.to_categorical(genfromtxt(LABEL_DIR, delimiter=','))
    return train_data, train_labels


def rebalanceData(x_train, y_train):
    ##Check class distribution
    storeAllocated = []
    x_out = np.empty((0,) + x_train.shape[1:])
    y_out = np.empty((0,4))
    [print("s" + str(s) + ":" +  str(len(np.argwhere(y_train[:,s] == 1)))) for s in range(0,4)]
    # storeAllocated = [len(np.argwhere(y_train[:,s] == 1)) for s in range(0,4)]
    # minNum = storeAllocated[3]
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

if __name__ == '__main__':
    
    model = initializeModelConv()    
    model.compile(optimizer='adam',#tf.keras.optimizers.Adam(lr=5e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])
    print(model.summary())
    print('This is WORKING_DIR')
    print(WORKING_DIR)
    
    filename = '/model_spectrogram_' + SESSION_ID + '_data.csv'
    train_data, train_labels = loadInputData(filename)
    win = 32
    #train_data = rolling_window(train_data.T,win).transpose(1,0,2)
    #train_labels = train_labels[0:-win+1,:]
    
    print('Here are the shapes of our dataset')
    print('Data: ' + str(train_data.shape))
    print('Labels: ' + str(train_labels.shape))
    print(train_data[:10,:])
    print(train_labels[:10,:])
    #train_data = normalize(train_data, axis=0, norm='l2')
    #train_data = scale(train_data,axis=0)
    acc, val_acc, loss,val_loss = [],[],[],[]
    #history = model.fit(np.expand_dims(train_data,3),train_labels, batch_size=900, epochs=100)
    #model.fit(train_data, train_labels, batch_size=7200, epochs=50, validation_split=0.2, verbose=1,)
    #(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    #print(train_labels.shape)
    # plt.show()
    ## Running k-folds classification to improve generalization and reduce overfitting
    for i in range(0,10):

        ##Only need to balance the training data, not the validation data.
        x_train, x_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=0.2, shuffle= True)
        x_train = np.expand_dims(rolling_window(x_train.T,win).transpose(1,0,2),3)
        y_train = y_train[0:-win+1,:]
        x_train = x_train[:-round(win/2),:]
        y_train = y_train[round(win/2):,:]
        x_train, y_train = rebalanceData(x_train, y_train) ##Utilize a form of downsampling rather than synthetic data
        
        x_valid = np.expand_dims(rolling_window(x_valid.T,win).transpose(1,0,2),3)
        y_valid = y_valid[0:-win+1,:]
        x_valid = x_valid[:-round(win/2),:]
        y_valid = y_valid[round(win/2):,:]
        history = model.fit(x_train, y_train, batch_size=900, epochs=50, validation_data=(x_valid, y_valid), verbose=1,)
        #acc.append(history.history['accuracy'])
        #val_acc.acc(history.history['val_accuracy'])
        #loss.append(history.history['loss'])
        #val_loss.append(history.history['val_loss'])
    
    print("Finished model fitting")
    today = datetime.now().strftime('%m_%d_%Y_%H-%M')
    saveName = WORKING_DIR+nameappend+ "_date-" + today+ "_model"
    print('Saving the model on ' + today + " to " + saveName)
    
    # model_json = model.to_json()
    # with open(saveName + ".json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights(saveName + ".h5") # serialize weights to HDF5
    print("Saved model to disk")
    model.save(saveName + "_iter0.h5")


    #print("Starting post-processing on single test dataset...this may take a while")
    #predictedHypnogram = model.predict(x_train)
    #newPredict, newProbVal = postProcessHypnogram(predictedHypnogram)
    #epochs = range(len(acc))
    #tempOrig = onehot_to_array(y_train)
    #tempTime = np.linspace(1/hypFs, len(tempOrig)/hypFs, len(tempOrig))
    #storeAcc, hFig = plotValidation(tempTime, tempOrig, newPredict) ## Plot both the original and DNN output

    
    print("Starting prediction...")
    train_data = rolling_window(train_data.T,win).transpose(1,0,2)
    train_labels = train_labels[0:-win+1,:]
    train_data = train_data[:-round(win/2),:]
    train_labels = train_labels[round(win/2):,:]
    hypnogramOrig = onehot_to_array(train_labels)
    hypTimeAxis = np.linspace(1/hypFs, len(hypnogramOrig)/hypFs, len(hypnogramOrig))
    predictedHypnogram = model.predict(np.expand_dims(train_data,3))
    # # serialize model to JSON
    predictedHypnogram = np.array(predictedHypnogram)
    # numpy.savetxt(WORKING_DIR + "/Hypnogram_" + SESSION_ID + "_" + nameappend + ".csv", a, delimiter=",")
    
    
    # # later...
    
    # # load json and create model
    # json_file = open(WORKING_DIR +nameappend+"_model.json", 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = keras.models.model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights(WORKING_DIR+nameappend+"_model.h5")
    # print("Loaded model from disk")
    print("Starting post-processing...this may take a while")
    newPredict, newProbVal = postProcessHypnogram(predictedHypnogram)
    
    epochs = range(len(acc))
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
    
    storeAcc, hFig = plotValidation(hypTimeAxis[:-win+1], hypnogramOrig[:-win+1], newPredict) ## Plot both the original and DNN output
