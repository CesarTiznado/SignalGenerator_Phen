#!/usr/bin/env python
# coding: utf-8

# ## Explorer Waveform

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import h5py as h5
import glob, os, sys
import copy

class Phen:
    def __init__(self,Path,Generator):
        self.sPath = Path
        self.sGenerator = Generator
        dataFile_Dir = Path
        
        rawdata_files = sorted(glob.glob(dataFile_Dir + '*.csv'))
        j=0
        print("*******************************")
        print("The available set of data are the following:")
        for i in range (0,len(rawdata_files)):
            #print(f"No.{j}", rawdata_files[i])
            j+=1
        print("The set segments of data available are the following:")
        print(len(rawdata_files))
        self.rawdata_files = rawdata_files
            
    def select_Data(self,segment):
        print("*******************************")
        print("Chosen strain data segments:")
        print(self.rawdata_files[segment])

        dataFile = pd.read_csv(self.rawdata_files[segment])
        self.dataFile = dataFile
        
        # self.dataFile = dataFile
        #
        print(f"{self.sGenerator} Keys: %s" % dataFile.keys())
        print("*******************************")
        return self.dataFile


# In[ ]:




