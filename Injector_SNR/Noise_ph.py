#!/usr/bin/env python
# coding: utf-8

#  # Noise reader

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import h5py as h5
import glob, os, sys
import copy


# ## Noise class

# In[2]:


class Noise:
    def __init__(self,Path,Detector):
        self.sPath = Path
        self.sDetector = Detector
        dataFile_Dir = Path
        
        rawdata_files = sorted(glob.glob(dataFile_Dir + '*-4096.hdf5'))
        j=0
        print("*******************************")
        print("The available sets of data are the following:")
        for i in rawdata_files:
            j+=1
            print(f"No.{j}", rawdata_files)
        self.rawdata_files = rawdata_files
    def select_Data(self,segment):
        print("Chosen strain data segment:")
       
        print(self.rawdata_files[segment])
        print("*******************************")
        dataFile = h5.File(self.rawdata_files[segment],"r")
        self.dataFile = dataFile
        
        # self.dataFile = dataFile
        #
        print(f"{self.sDetector} Keys: %s" % dataFile.keys())
        print("*******************************")
        return self.dataFile
    
    
    
    def Gps_duration(self):
        self.meta = self.rawdata_files['meta']
        metaKeys  = self.meta.keys()
        gpsStart  = self.meta['GPSstart'][()]
        duration  = self.meta['Duration'][()]
        gpsEnd    = gpsStart + duration

        print("GPS start time: %s" % gpsStart)
        print("Segment Duration (s):", duration)
        
        return gpsStart,duration

class Noise_strain:
    def __init__(self,dataFile):
        self.dataFile = dataFile
    def strain(self):
        strain = self.dataFile['strain']['Strain'][()]
        print("*******************************")
        print("Strain_signal:",strain)
        return strain
    def strain_time(self):
        self.meta = self.dataFile['meta']
        ts = self.dataFile['strain']['Strain'].attrs['Xspacing']
        metaKeys  = self.meta.keys()
        gpsStart  = self.meta['GPSstart'][()]
        duration  = self.meta['Duration'][()]
        gpsEnd    = gpsStart + duration
        time_strain = np.arange(gpsStart, gpsEnd, ts)
        print("*******************************")
        print("GPS start time: %s" % gpsStart)
        print("Segment Duration (s):", duration)
        print("Strain_time: %s" % time_strain)

        return gpsStart,duration,time_strain

# ## Injection

# In[6]:


# import random
# fs=16384
# len_Segmento = fs * 16 
# segments = []
# n = int(len(strain_time_L1)/len_Segmento )
# jiter = random.randint(0,fs)

# print(jiter)
# h_phen = selected_Ph["H"]
# h_time = selected_Ph["Time"]
# h_new = copy.copy(h_phen)
# strain_new = copy.copy(strain_data_L1)



# # Reamplitude of the phenomenological wave
# factor = np.max(strain_data_L1)
# h_new = h_new * factor


# # In[7]:


# get_ipython().run_cell_magic('time', '', 'labels = []\nfor i in range(0,n):\n    s = i*16*fs\n    labels.append(s)\n    #print(f"Bloque:{i}", s)\n    for k in range(0,len(h_new)):\n        strain_new[s + k] = h_new[k] + strain_data_L1[s + k] \n#print(f"Etiquetas de los tiempos:\\n {labels}")')


# # In[9]:


# get_ipython().run_cell_magic('time', '', 'h_try = copy.copy(h_phen)\nstrain_try_n = copy.copy(strain_data_L1)\njit = []\njit_stor = []\nD = 256\njit_stor = np.zeros(D)\n\nfactor =     abs(np.max(strain_data_L1)) - abs(np.min(strain_data_L1)) / 2 \nh_try = h_try * factor\n\n\njit_stor[0] = random.randint(0,2*fs)\n\ndb = True\ncontador = 0\ni=1\nwhile db==True:\n    jit_stor[i] = random.randint(-2*fs,2*fs)\n    jit_stor[i]= int(jit_stor[i-1] + len(h_try) +16*fs + jit_stor[i] )\n    #print(jit_stor[i])\n    if jit_stor[i] >= len(strain_data_L1):    \n        db = False\n        jit_stor = jit_stor[jit_stor!=0]\n    else:\n        for j in range(0,len(h_try)):\n            k = int(jit_stor[i-1])\n            strain_try_n[k + j] =  strain_data_L1[k+j] + h_try[j]\n    i+=1\n\n# An Alternative\n# for i in range(1,D):\n#     jit_stor[i] = random.randint(-2*fs,2*fs)\n#     jit_stor[i]= int(jit_stor[i-1] + len(h_try) +16*fs + jit_stor[i] )\n#     if jit_stor[i]>= len(strain_data_L1):\n#        jit_stor = jit_stor[jit_stor!=0]\n#        break\n#     else:\n#         #print(jit_stor[i])\n#         # jit.append(G)\n#         # print(jit[i])\n#         for j in range(0,len(h_try)):\n#             k = int(jit_stor[i-1])\n#             strain_try_n[k + j] =  strain_data_L1[k+j] + h_try[j]\n    \n#print(jit_stor)\n\n# h_labeled = np.zeros(len(jit_stor))\n# j = 0\n# for i in jit_stor:\n#     h_labeled[j] = strain_try_n[i]\n#     j+=1')


# # ## SNR
# # 
# # The equation that is used in the SNR implementation came from,** https://iopscience.iop.org/article/10.1088/2632-2153/ab7d31/pdf.**
# # 
# # 
# # 

# # In[11]:


# get_ipython().run_cell_magic('time', '', 'from scipy.fft import fft\n# \n#https://iopscience.iop.org/article/10.1088/2632-2153/ab7d31/pdf\n# First we calculate de FFT of h(t)\n# h(t) -> h(f)\n\nh_f  = fft(strain_try_n)\n# Complex Conjugate\nh_fcon = np.conj(h_f) \n\nprint(f"Fourier Transform of h(t), h(f): {h_f}")\nprint(f"Complex conjugate {h_fcon}")')



 


# In[16]:


#  We need that the values share the same length.
# Therefore, we should had a df to increment the frequency between the boundaries.
# The result of the SNR is a Scalar, because sum all the littles steps carry on by the integral, in this case a small diferences.






