# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 18:31:54 2020

@author: danny
"""

import numpy as np 
from statistics import mode
import collections
from functions.helper_functions import * 
# from main import CONFIG_PATH

# config = load_config(CONFIG_PATH / "my_config.yaml")
# globals().update(config)
## Take the max of a rolling average of probabilities to smooth out the values
def rollingPredictionAvg(probVal, predictedHypnogram, winsize, hypFs = 2):
    numTimes  = len(probVal)
    newPredict = np.zeros(numTimes)
    newProbVal = np.zeros(numTimes)
    
    ## Perform the rolling average
    for i in range(winsize+1, numTimes): 
        counter = collections.Counter(predictedHypnogram[i-winsize:i])
        curVals = counter.most_common(1)[0][0]
        newProbVal[round(i-winsize/hypFs)] = np.mean(probVal[i-winsize:i]);
        newPredict[round(i-winsize/hypFs)] = curVals;
    
    return newPredict, newProbVal

def getTransitionTimes(curArray):
   ##Transition times with a boolean vector
   ##Return a Mx2 array where 
   ##Column 1 is startTime and Column 2 is endTime
   transitions = np.array([[0,0]])
   prevIndex = 0;
   curIndex = 0;
   prevState = 0;
   curState = 0;
   for i in range(0,len(curArray)):
       if(curArray[i] != curState):
           prevState = curState
           curState = curArray[i]
           prevIndex = curIndex
           curIndex = i             
           transitions = np.vstack((transitions, [prevIndex, curIndex]))
                
           
   return transitions
   


def removeShortTransitions(predictedHypnogram):
    
    ## Removing epochs with shorter than 30s epochs, and filling in with epochs around it
    ## Repeating twice
    for numRuns in range(0,2):
        tTimes = getTransitionTimes(predictedHypnogram)
        for i in range(0, len(tTimes)):
            if(np.diff(tTimes[i,:]) < 20):
                predictedHypnogram[tTimes[i,0]:tTimes[i,1]] = np.nan
                
        predictedHypnogram = fill(predictedHypnogram)
    return predictedHypnogram

def postProcessHypnogram(predictedLabels, min_accuracy=0.9, winsize=10):

    # hypnoOutput = xlsread(['G:\sleep-DNN\data\Hypnogram_02-25_spectrogram_toHour28_20valsplit_40fold_Dropout0p3_simpler.csv']); %%This baby achieved 94% accuracy
    # [probVal, predictHyp] = max(predictedLabels);
    newPredict, newProbVal = np.argmax(predictedLabels, axis = 1), np.max(predictedLabels, axis=1)
    newPredict, newProbVal = rollingPredictionAvg(newProbVal, newPredict, winsize)
    rawPredict = newPredict
    newPredict[newProbVal < min_accuracy] = np.nan ##Then, try to filter out any probabilities less than 95%
    newPredict = fill(newPredict)
    newPredict = removeShortTransitions(newPredict)
    newPredict, newProbVal = rollingPredictionAvg(newProbVal, newPredict, winsize)
    
    return newPredict, newProbVal
  
    
if __name__ == '__main__':
    print("In here")
