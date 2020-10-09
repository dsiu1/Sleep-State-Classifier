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
global CONFIG_PATH, SESSION_ID,DATA_DIR, compareHypnograms
CONFIG_PATH = Path(__file__).parent.absolute() 
print(CONFIG_PATH)
os.chdir(CONFIG_PATH)
CONFIG_PATH = CONFIG_PATH /  "config"

# Load new test data
config = load_config(CONFIG_PATH / "my_config.yaml")
globals().update(config)

### Will need to refactor the code to make dependencies less confusing. Hacky way to call these functions for now 
import DNN.sleep_wake_classifier_predict,  DNN.sleep_wake_classifier_simpler, DNN.sleep_wake_classifier_retrain
## MATLAB calls this script externally, setting the working_dir and session 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processing path inputs from external call")
    parser.add_argument("-w", "--working_dir", help="Changing the working dir", nargs="?", const=1, default=WORKING_DIR)
    parser.add_argument("-s", "--session", help="Session ID within the working dir", nargs="?",const=1, default=SESSION_ID)
    args = parser.parse_args()
    
    WORKING_DIR = args.working_dir
    SESSION_ID = args.session
    print(args)
	


    hFig = DNN.sleep_wake_classifier_predict.run(WORKING_DIR=WORKING_DIR,SESSION_ID=SESSION_ID)
    
    # sleep_wake_classifier_simpler.run()
    # sleep_wake_classifier_retrain.run()
    # 

