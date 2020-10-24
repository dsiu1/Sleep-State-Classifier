# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 10:50:56 2020

@author: danny
"""

from functions.helper_functions import *
from pathlib import Path
import matplotlib.pyplot as plt
import os
import argparse
from DNN.sleep_wake_classifier import SleepClassifier
global CONFIG_PATH, SESSION_ID,DATA_DIR, compareHypnograms
CONFIG_PATH = Path(__file__).parent.absolute() 
print(CONFIG_PATH)
os.chdir(CONFIG_PATH)
CONFIG_PATH = CONFIG_PATH /  "config"

# Load new test data
config = load_config(CONFIG_PATH / "my_config.yaml")
globals().update(config)


## MATLAB calls this script externally, setting the working_dir and session 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processing path inputs from external call")
    parser.add_argument("-w", "--working_dir", help="Changing the working dir", nargs="?", const=1, default=WORKING_DIR)
    parser.add_argument("-s", "--session", help="Session ID within the working dir", nargs="?",const=1, default=SESSION_ID)
    args = parser.parse_args()
    
    WORKING_DIR = args.working_dir
    SESSION_ID = args.session
    print(args)
    sleepModel = SleepClassifier(CONFIG_PATH,WORKING_DIR,SESSION_ID)
    # SWC.fit()
    sleepModel.predict()
    
    # If retraining or fitting for the first time, use the fit() method
    


