#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import eppy as ep
from eppy import modeleditor
import sys
from eppy.modeleditor import IDF
import pandas as pd
from statistics import mean
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import timeit
import time
#import multiprocessing
from eppy.pytest_helpers import do_integration_tests
from eppy.runner.run_functions import install_paths, EnergyPlusRunError
from eppy.runner.run_functions import multirunner
from eppy.runner.run_functions import run
from eppy.runner.run_functions import runIDFs
import zeppy
from zeppy import ppipes
import geneticalgs as gas


# In[2]:


path ="/speed-scratch/z_khoras"


# In[5]:


iiddfile = path+ '/EP-8-9/EnergyPlus-8-9-0/Energy+.idd'
IDF.setiddname(iddfile)


# In[9]:


"""multiprocessing runs"""

# using generators instead of a list
# when you are running a 100 files you have to use generators

import os 
from eppy.modeleditor import IDF
from eppy.runner.run_functions import runIDFs

def make_options(idf):
    idfversion = idf.idfobjects['version'][0].Version_Identifier.split('.')
    idfversion.extend([0] * (3 - len(idfversion)))
    idfversionstr = '-'.join([str(item) for item in idfversion])
    fname = idf.idfname
    options = {
        'ep_version':idfversionstr,
        'output_prefix':os.path.basename(fname).split('.')[0],
        'output_suffix':'C',
        'output_directory':os.path.dirname(fname),
        'readvars':True,
        'expandobjects':True
        }
    return options




def main(X):
    iddfile = path+ '/EP-8-9/EnergyPlus-8-9-0/Energy+.idd'
    IDF.setiddname(iddfile)
    epwfile = path+'/EP-Input/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
    w = X.item()
    #w = np.asscalar(X)
    fname1 = path +'/EP-Input/sTestFor8.9.idf'
    epwfile = path+'/EP-Input/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
    idf = IDF(fname1,epwfile)
    occlightingmodel = idf.idfobjects['EnergyManagementsystem:program'][1]
    occlightingmodel.Program_Line_7 ="set x=" + str(w) #maybe we can use directly "set x=" + str(X[0])
    fnames=[]
    for i in range (1,5):
        idf.saveas(path +'/EP-Input/s%dTestFor8.9.idf'%(i))
        #globals()["idfname{}".format(i)] = '/Users/Mehdi/Downloads/oooooof/EPPY/GA/practice%d.idf'%(i)
        fnames.append(path +'/EP-Input/s%dTestFor8.9.idf'%(i))
        #files = os.listdir(path)
        #fnames = [f for f in files if f[-4:] == '.idf']
    idfs = (IDF(fname, epwfile) for fname in fnames)
    runs = ((idf, make_options(idf) ) for idf in idfs)
    num_CPUs = 16
    runIDFs(runs, num_CPUs)
    TRELC=[]

    for i in range (1,5):
        Data=pd.read_csv('practice%d.csv'%(i))
        ELC=Data['LIGHT:Lights Electric Energy [J](TimeStep)'].sum()*2.78*10**(-7)
        TRELC.append(ELC)
    return np.sum(TRELC)

#import time
#starttime = time.time()
#if __name__ == '__main__':
    
    
#    main()
#print('Time taken = {} seconds'.format(time.time() - starttime))
    


# In[10]:


algorithm_param = {'max_num_iteration': 5,'population_size':3,'mutation_probability':0.35,'elit_ratio': 0.05,'crossover_probability': 0.7,'parents_portion': 0.3,'crossover_type':'uniform','max_iteration_without_improv':None}


# In[11]:


varbound= np.array([[1,2]])
#np.array([[1,5]])

model=ga(function=main,dimension=1,variable_type='int',variable_boundaries=varbound, function_timeout=20000,
         algorithm_parameters=algorithm_param)
model.run()


# In[ ]:




