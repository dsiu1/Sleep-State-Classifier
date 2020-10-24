# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 08:19:02 2020

@author: danny

Purpose: Turn the sleep wake classifier into a Class for simpler training, predictions, and retraining
"""
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
import seaborn as sns

from functions.hypnogram_postprocess import postProcessHypnogram
from functions.helper_functions import *
from sklearn.preprocessing import normalize
from operator import itemgetter

class SleepClassifier():
    def __init__(self,config,WORKING_DIR,SESSION_ID):
        self.CONFIG_PATH = config
        self.WORKING_DIR = WORKING_DIR
        self.SESSION_ID = SESSION_ID
        config = load_config(config / "my_config.yaml")
        globals().update(config)
            
            
        
    def initializeModel(self):
    
        
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
    def loadInputData(self,filename):
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
    def rebalanceData(self,x_train, y_train):
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
    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    def my_learning_rate(self, epoch):
        print(epoch)
        return 1e-2/(epoch/2+1)
    
    
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
    def fit(self):
        WORKING_DIR = self.WORKING_DIR
        SESSION_ID = self.SESSION_ID
        model = self.initializeModel()    
        model.compile(optimizer='adam',#tf.keras.optimizers.Adam(lr=1e-2),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'categorical_accuracy'])
        lrs = keras.callbacks.LearningRateScheduler(self.my_learning_rate)
        print(model.summary())
        print('This is WORKING_DIR')
        print(WORKING_DIR)
        
        filename = '/model_spectrogram_' + SESSION_ID + '_data.csv'
        train_data, train_labels = self.loadInputData(filename)
        samples = int(np.round(train_labels.shape[0]*0.8))
        test_data = train_data[samples:,:]
        test_labels = train_labels[samples:,:]
        train_data = train_data[0:samples,:]
        train_labels = train_labels[0:samples,:]
        
        hypnogramOrig = onehot_to_array(test_labels)
        hypTimeAxis = np.linspace(1/hypFs, len(hypnogramOrig)/hypFs, len(hypnogramOrig))
        print('Here are the shapes of our dataset')
        print('Data: ' + str(train_data.shape))
        print('Labels: ' + str(train_labels.shape))
        print(train_data[:10,:])
        print(train_labels[:10,:])
        
        acc, val_acc, loss,val_loss = [],[],[],[]
        
    
        ## Running k-folds classification to improve generalization and reduce overfitting
        for i in range(0,5):
        
            ##Only need to balance the training data, not the validation data.
            x_train, x_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=0.2, shuffle= True)
            x_train, y_train = self.rebalanceData(x_train, y_train) ##Utilize a form of downsampling rather than synthetic data
            history = model.fit(x_train, y_train, batch_size=900, epochs=100, validation_data=(x_valid, y_valid), verbose=1)#, callbacks=[lrs])
    		
            # acc.append(history.history['accuracy'])
            # val_acc.acc(history.history['val_accuracy'])
            acc.append(history.history['acc'])
            val_acc.append(history.history['val_acc'])
            loss.append(history.history['loss'])
            val_loss.append(history.history['val_loss'])
        
        print("Finished model fitting")
        acc_all = np.array(acc).flatten().tolist()
        valacc_all = np.array(val_acc).flatten().tolist()
        loss_all = np.array(loss).flatten().tolist()
        valloss_all = np.array(val_loss).flatten().tolist()
        epochs = range(len(acc_all))
            
        print("Starting prediction...")
        predictedHypnogram = model.predict(test_data)
        # Serialize model to JSON
        predictedHypnogram = np.array(predictedHypnogram)
        
        today = datetime.now().strftime('%m_%d_%Y_%H-%M')
        saveName = WORKING_DIR+nameappend+ "_date-" + today+ "_model"
        print('Saving the model on ' + today + " to " + saveName)
        model.save(saveName + "_iter0.h5")
        print("Saved model to disk")
        
        print("Starting post-processing...this may take a while")
        newPredict, newProbVal = postProcessHypnogram(predictedHypnogram)
    	
    	
        # Plot evaluations
        plt.figure()
        plt.subplot(211)
        plt.plot(epochs, acc_all, 'r', label='Training accuracy')
        plt.plot(epochs, valacc_all, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)
        plt.subplot(212)
        plt.plot(epochs, loss_all, 'r', label='Training Loss')
        plt.plot(epochs, valloss_all, 'b', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.legend(loc=0)
        plt.show()
        
    	# Plot predicted hypnogram against the ground truth
        storeAcc, hFig, accuracy, F1 = plotValidation(hypTimeAxis, hypnogramOrig, newPredict) ## Plot both the original and DNN output
        df = pd.DataFrame(storeAcc)
        df.columns = ["Awake", "Unc", "NREM", "REM"]
        df.index = ["Awake", "Unc", "NREM", "REM"]
        plt.figure()
        ax = sns.heatmap(df, annot=True, cmap='Blues')
        ax.set(xlabel="Actual", ylabel = "Predicted")
        print("Accuracy: {}   F1 score: {}".format(round(accuracy,2), round(F1,2)))    
            
        
        
    def predict(self):
        WORKING_DIR = self.WORKING_DIR
        SESSION_ID = self.SESSION_ID
        DATA_DIR = '/model_spectrogram_' + SESSION_ID +'_data.csv'
        if compareHypnograms:
            train_data, train_labels = self.loadInputData(DATA_DIR)
        else:
            train_data = pd.read_csv(WORKING_DIR + DATA_DIR, header=None, delimiter=',').to_numpy()
        
        #train_data = normalize(train_data, axis=1, norm='l2')
        #train_data = scale(train_data,axis=0)
        loaded_model = keras.models.load_model(currentModel + ".h5")
        print("Loaded model from disk " + currentModel)
        
        
        ## CNN predictions
        if modelCNN:
            win = 32
            train_data = self.rolling_window(train_data.T,win).transpose(1,0,2)
            train_labels = train_labels[0:-win+1,:]
            train_data = train_data[:-round(win/2),:]
            train_labels = train_labels[round(win/2):,:]
            predictedHypnogram = loaded_model.predict(np.expand_dims(train_data,3))
      
        else:
            predictedHypnogram = loaded_model.predict(train_data)
            newPredict, newProbVal = postProcessHypnogram(predictedHypnogram, 0.95)
            hypTimeAxis = np.linspace(1/hypFs, len(newPredict)/hypFs, len(newPredict))
        
        if compareHypnograms:
            hypnogramOrig = onehot_to_array(train_labels)
            storeAcc, hFig,accuracy,F1 = plotValidation(hypTimeAxis, hypnogramOrig, newPredict) ## Plot both the original and DNN output
            df = pd.DataFrame(storeAcc)
            df.columns = ["Awake", "Unc", "NREM", "REM"]
            df.index = ["Awake", "Unc", "NREM", "REM"]
            ax = sns.heatmap(df, annot=True, cmap='Blues')
            ax.set(xlabel="Actual", ylabel = "Predicted")
            print("Accuracy: {}   F1 score: {}".format(round(accuracy,2), round(F1,2)))
            # print(np.round(df,2))
            return(hFig)
    
            
        
        
        a = numpy.asarray(newPredict)
        numpy.savetxt(WORKING_DIR + "/Hypnogram_" +SESSION_ID+ ".csv", a, delimiter=",")
        
        ## To Matfile. Need to include these variables to match existing format
        staticIndices = np.zeros(len(newPredict))
        NREMthresh = 1
        REMthresh = 1
        savemat(WORKING_DIR + "/Hypnogram_" +SESSION_ID+ ".mat", mdict={'hypnogram':a, 
                                                                      'hypTimeAxis':hypTimeAxis, 
                                                                      'staticIndices':staticIndices, 
                                                                      'NREMthresh': NREMthresh, 
                                                                      'REMthresh':REMthresh})
            
            
                
                
                
                
        
        
        
        
