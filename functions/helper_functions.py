# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 09:41:45 2020

@author: danny
"""
import numpy as np
from scipy import ndimage as nd
import yaml
import matplotlib.pyplot as plt
def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    #import numpy as np
    #import scipy.ndimage as nd

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]
def onehot_to_array(hypnogram):
    rows, cols = hypnogram.shape
    numLabels  = rows if rows < cols else cols
    axisSum = 1 if rows > cols else 0
    hypnogram_array = np.sum(hypnogram*np.array([i for i in range(1,numLabels+1)]),axis=axisSum)-1
    return hypnogram_array

# Function to load yaml configuration file
def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    
    config["winsize"] = config["winsize"]*config["hypFs"]
    config["nameappend"] = config["nameappend"] + config["SESSION_ID"]
    return config

def plotValidation(hypTimeAxis, hypnogramOrig, newPredict):
    hFig = plt.figure()
    plt.subplot(211)
    plt.plot(hypTimeAxis, hypnogramOrig)
    # plt.xlim(0, 3600)
    plt.subplot(212)
    plt.plot(hypTimeAxis, hypnogramOrig, 'r', label='Ground truth hypnogram')
    plt.plot(hypTimeAxis, newPredict, 'b', label='DNN Hypnogram')
    plt.title('Hypnogram Training Results')
    plt.legend(loc=0)
    # plt.xlim(0, 3600)
    plt.show()
    
    storeAccRaw = [];
    storeAccPost = np.zeros((4,4));
    # rawPredictAdj = rawPredict-1;
    for s1 in range(0,4):
        for s2 in range(0,4):
            getStateLength = len(np.argwhere(hypnogramOrig == s1));
            # rawPredictAcc = length(find(hypnogram(1:201600) == s1-1 & rawPredictAdj == s2-1))/getStateLength;
            if(getStateLength == 0):
                postPredictAcc = 0
            else:
                postPredictAcc = len(np.argwhere(np.logical_and(hypnogramOrig == s1, newPredict == s2)))/getStateLength;
            # storeAccRaw(s1,s2) = rawPredictAcc;
            storeAccPost[s1,s2] = postPredictAcc;
    accuracy = np.sum(np.diagonal(storeAccPost))/storeAccPost.sum()
    F1 = np.mean([2*storeAccPost[i,i]/(np.sum(storeAccPost[i,:])  + np.sum(storeAccPost[:,i])) for i in range(0,4)])
                                   
    print(np.round(storeAccPost,2))
    return storeAccPost, hFig, accuracy, F1



## Feed in data to the neural network that has already been windowed
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

