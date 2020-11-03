#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import eppy as ep
from eppy import modeleditor
#import sys
from eppy.modeleditor import IDF
import pandas as pd
from statistics import mean
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
#import timeit
#import time
from eppy.pytest_helpers import do_integration_tests
from eppy.runner.run_functions import install_paths, EnergyPlusRunError
from eppy.runner.run_functions import multirunner
from eppy.runner.run_functions import run
from eppy.runner.run_functions import runIDFs
#import zeppy
#from zeppy import ppipes
#import geneticalgs as gas


# In[ ]:


path ="/speed-scratch/z_khoras"


# In[ ]:


#pathnameto_eppy = '../'
#sys.path.append(pathnameto_eppy)


# In[ ]:


iddfile = path+ '/EP-8-9/EnergyPlus-8-9-0/Energy+.idd'
IDF.setiddname(iddfile)


# In[ ]:


#w = X.item()
#w = np.asscalar(X)
fname1 = path +'/EP-Input/sTestFor8.9.idf'
epwfile = path+'/EP-Input/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
idf = IDF(fname1,epwfile)
#occlightingmodel = idf.idfobjects['EnergyManagementsystem:program'][1]
#occlightingmodel.Program_Line_7 ="set x=" + str(w) #maybe we can use directly "set x=" + str(X[0])
fnames=[]
for i in range (1,5):
    idf.saveas(path +'/EP-Input/s%dTestFor8.9.idf'%(i))
    #globals()["idfname{}".format(i)] = '/Users/Mehdi/Downloads/oooooof/EPPY/GA/practice%d.idf'%(i)
    fnames.append(path +'/EP-Input/s%dTestFor8.9.idf'%(i))
    #files = os.listdir(path)
    #fnames = [f for f in files if f[-4:] == '.idf']
idfs = (IDF(fname, epwfile) for fname in fnames)
runs = ((idf, make_options(idf) ) for idf in idfs)
runIDFs(runs, num_CPUs)}


# In[ ]:




