##Running for Python 3.6
##Generating keras modeling to predict sleep-wake classification

import tensorflow as tf
from tensorflow import keras
import numpy
import pandas as pd
from numpy import genfromtxt
from sklearn.model_selection import train_test_split ##Need to randomly split data
from hypnogram_postprocess import postProcessHypnogram
from helper_functions import *
from sklearn.preprocessing import normalize, scale
from scipy.io import savemat
# globals().update(config)
from main import CONFIG_PATH
config = load_config(CONFIG_PATH / "my_config.yaml")
globals().update(config)


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

def run(WORKING_DIR=WORKING_DIR, SESSION_ID=SESSION_ID):
    DATA_DIR = '/model_spectrogram_' + SESSION_ID +'_data.csv'
    if compareHypnograms:
        train_data, train_labels = loadInputData(DATA_DIR)
    else:
        train_data = pd.read_csv(WORKING_DIR + DATA_DIR, header=None, delimiter=',').to_numpy()
    
    #train_data = normalize(train_data, axis=1, norm='l2')
    #train_data = scale(train_data,axis=0)
    # load json and create model
    # json_file = open(currentModel + ".json", 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model = keras.models.load_model(currentModel + ".h5")
    print("Loaded model from disk " + currentModel)
    win = 32
    
    ## CNN predictions
    # train_data = rolling_window(train_data.T,win).transpose(1,0,2)
    # train_labels = train_labels[0:-win+1,:]
    # train_data = train_data[:-round(win/2),:]
    # train_labels = train_labels[round(win/2):,:]
    # predictedHypnogram = loaded_model.predict(np.expand_dims(train_data,3))
    
    
    predictedHypnogram = loaded_model.predict(train_data)
    newPredict, newProbVal = postProcessHypnogram(predictedHypnogram, 0.95)
    hypTimeAxis = np.linspace(1/hypFs, len(newPredict)/hypFs, len(newPredict))
    
    # plt.figure()
    # plt.ion()
    # plt.subplot(211)
    # plt.pcolormesh(hypTimeAxis[:30000], np.linspace(0,40,train_data.shape[1]), train_data[:30000,:].T, shading='auto')
    # plt.colorbar()
    # plt.subplot(212)
    # plt.pcolormesh(hypTimeAxis[:30000], np.linspace(0,40,train_data.shape[1]), normalize(train_data[:30000,:], axis=1, norm='l2').T, shading='auto')
    # plt.colorbar()
    if compareHypnograms:
        hypnogramOrig = onehot_to_array(train_labels)
        # hypTimeAxis = hypTimeAxis[12000:48000]
        # hypnogramOrig = hypnogramOrig[12000:48000]
        # newPredict = newPredict[12000:48000]
        storeAcc, hFig = plotValidation(hypTimeAxis, hypnogramOrig, newPredict) ## Plot both the original and DNN output
        df = pd.DataFrame(storeAcc)
        df.columns = ["Awake P", "Unc P", "NREM P", "REM P"]
        df.index = ["Awake N", "Unc N", "NREM N", "REM N"]
        print(np.round(df,2))
        return(hFig)

        
    
    
    a = numpy.asarray(newPredict)
    numpy.savetxt(WORKING_DIR + "/Hypnogram_" +SESSION_ID+ ".csv", a, delimiter=",")
    
    ## To Matfile
    staticIndices = np.zeros(len(newPredict))
    NREMthresh = 1
    REMthresh = 1
    savemat(WORKING_DIR + "/Hypnogram_" +SESSION_ID+ ".mat", mdict={'hypnogram':a, 
                                                                  'hypTimeAxis':hypTimeAxis, 
                                                                  'staticIndices':staticIndices, 
                                                                  'NREMthresh': NREMthresh, 
                                                                  'REMthresh':REMthresh})
if __name__ == '__main__':
    compareHypnograms =1
    DATA_DIR = '/model_spectrogram_' + SESSION_ID +'_data.csv'
    if compareHypnograms:
        train_data, train_labels = loadInputData(DATA_DIR)
    else:
        train_data = pd.read_csv(WORKING_DIR + DATA_DIR, header=None, delimiter=',').to_numpy()
    
    #train_data = normalize(train_data, axis=1, norm='l2')
    #train_data = scale(train_data,axis=0)
    # load json and create model
    # json_file = open(currentModel + ".json", 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model = keras.models.load_model(currentModel + ".h5")
    print("Loaded model from disk " + currentModel)
    win = 32
    
    ## CNN predictions
    # train_data = rolling_window(train_data.T,win).transpose(1,0,2)
    # train_labels = train_labels[0:-win+1,:]
    # train_data = train_data[:-round(win/2),:]
    # train_labels = train_labels[round(win/2):,:]
    # predictedHypnogram = loaded_model.predict(np.expand_dims(train_data,3))
    
    
    predictedHypnogram = loaded_model.predict(train_data)
    newPredict, newProbVal = postProcessHypnogram(predictedHypnogram, 0.95)
    hypTimeAxis = np.linspace(1/hypFs, len(newPredict)/hypFs, len(newPredict))
    
    # plt.figure()
    # plt.ion()
    # plt.subplot(211)
    # plt.pcolormesh(hypTimeAxis[:30000], np.linspace(0,40,train_data.shape[1]), train_data[:30000,:].T, shading='auto')
    # plt.colorbar()
    # plt.subplot(212)
    # plt.pcolormesh(hypTimeAxis[:30000], np.linspace(0,40,train_data.shape[1]), normalize(train_data[:30000,:], axis=1, norm='l2').T, shading='auto')
    # plt.colorbar()
    if compareHypnograms:
        hypnogramOrig = onehot_to_array(train_labels)
        # hypTimeAxis = hypTimeAxis[12000:48000]
        # hypnogramOrig = hypnogramOrig[12000:48000]
        # newPredict = newPredict[12000:48000]
        storeAcc, hFig, totalAcc, F1 = plotValidation(hypTimeAxis, hypnogramOrig, newPredict) ## Plot both the original and DNN output
        df = pd.DataFrame(storeAcc)
        df.columns = ["Awake P", "Unc P", "NREM P", "REM P"]
        df.index = ["Awake N", "Unc N", "NREM N", "REM N"]
        print(np.round(df,2))

        