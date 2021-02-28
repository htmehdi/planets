#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import eppy as ep
from eppy import modeleditor
import sys
from eppy.modeleditor import IDF
import pandas as pd
import csv
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


# In[2]:


path = "/Users/15142/Desktop/IBPS/BGT/"


# In[3]:


iddfile = "/EnergyPlusV8-9-0/Energy+.idd"
IDF.setiddname(iddfile)


# In[7]:


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
    from eppy.modeleditor import IDF
    iddfile = "/EnergyPlusV8-9-0/Energy+.idd"
    IDF.setiddname(iddfile)
    epwfile = "/Users/15142/Desktop/total-3OCCs-8.9/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw"
    #mapping for heating maximum threshold 
    maxh=np.arange(22,30.1,0.5)
    a = maxh[int(X[0])]
    # mapping for heating minimum threshold
    minhandc=np.arange(15,20.1,0.5)
    b= minhandc[int(X[1])]
    #mapping for cooling maximum threshold
    maxc=np.arange(25,30.1,0.5)
    c=maxc[int(X[2])]
    #mapping for colling minimum threshold
    d=minhandc[int(X[3])]
    #mapping for U-factor glazing
    e1=np.arange(1.4,2.2,0.1)
    e2=e1[int(X[4])]
    #mapping for SHGC
    SHGC=np.arange(0.3,0.61,0.05)
    f=SHGC[int(X[5])]
    # mapping axis to an array
    g1=[0,90,180,270]
    g2=g1[int(X[6])]
    #mapping for blind solar transmittance
    Blind=np.arange(0.05,0.2,0.05)
    h=Blind[int(X[7])]
    #mapping wwr
    i1=np.arange(0.2,0.71,0.05)
    i2=i1[int(X[8])]
    #mapping for roof R-value
    j1=np.arange(0.55,1.26,0.1)
    j2=j1[int(X[9])]
    #mapping for floor R-value
    k1=np.arange(0.13,0.54,0.1)
    k2=k1[int(X[10])]
    #mapping for interior wall R-value
    
    #l1=w[int(X[11])]
    #mapping for exterior wall R-value
    l1=np.arange(0.4,1.2,0.1)
    l2=l1[int(X[11])]
    
    fname1 = "/Users/15142/Desktop/IBPS/BGT/BGT.idf"
    epwfile = "/Users/15142/Desktop/total-3OCCs-8.9/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw"
    idf = IDF(fname1,epwfile)
    occthermostatmodel = idf.idfobjects['EnergyManagementsystem:program'][8]
    occthermostatmodel.Program_Line_72 ="IF HSP < " + str(b) #maybe we can use directly "set x=" + str(X[0])
    occthermostatmodel.Program_Line_73 ="set HSP = " + str(b)
    # the second variable:
    occthermostatmodel.Program_Line_74="ELSEIF HSP >" + str(a)
    occthermostatmodel.Program_Line_75="set HSP = " + str(a)
    # the third variable:
    occthermostatmodel.Program_Line_78 ="IF CSP < " + str(d)
    occthermostatmodel.Program_Line_79 ="set CSP = " + str(d)
    # the forth variable
    occthermostatmodel.Program_Line_80 ="ELSEIF CSP > " + str(c)
    occthermostatmodel.Program_Line_81 ="set CSP = " + str(c)
    # u-factor glazing
    Windowmaterial = idf.idfobjects['WindowMaterial:SimpleGlazingSystem'][0]
    Windowmaterial.UFactor=e2
    #SHGC
    Windowmaterial.Solar_Heat_Gain_Coefficient=f
    #the north axes
    office=idf.idfobjects['Building'][0]
    office.North_Axis=g2
    #blind solar Transmittance
    Blind_material=idf.idfobjects['WindowMaterial:Shade'][0]
    Blind_material.Solar_Transmittance=h
    #floor R-value
    F16_acoustic_tile_floor=idf.idfobjects['Material'][13]
    F16_acoustic_tile_floor.Thickness=k2
    #roof R-value
    F16_acoustic_tile_roof=idf.idfobjects['Material'][5]
    F16_acoustic_tile_roof.Thickness=j2
    # interior wall R-value
    #G01a_19mm_gypsum_board=idf.idfobjects['Material'][2]
    #G01a_19mm_gypsum_board.Thickness=l1
    #exterior wall R-value
    halfinch_gypsum=idf.idfobjects['Material'][6]
    halfinch_gypsum.Thickness=l2
    idf.saveas('/Users/15142/Desktop/IBPS/BGT/BGT.idf')
  
    #WWr
    from geomeppy import IDF
    #iddfile = "/EnergyPlusV8-9-0/Energy+.idd"
    #IDF.setiddname(iddfile)
    fname2 = "/Users/15142/Desktop/IBPS/BGT/BGT.idf"
    idf1 = IDF(fname2,epwfile)
    idf1.set_wwr(wwr=0, wwr_map={180: i2}, force=True, construction= "Exterior Window")
    idf1.saveas('/Users/15142/Desktop/IBPS/BGT/BGT.idf')
    #setting wshCTRL
    from eppy.modeleditor import IDF
    fname1 = "/Users/15142/Desktop/IBPS/BGT/BGT.idf"
    epwfile = "/Users/15142/Desktop/total-3OCCs-8.9/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw"
    idf = IDF(fname1,epwfile)
    sub_surface = idf.idfobjects['FenestrationSurface:Detailed'][0]
    sub_surface.Shading_Control_Name="wshCTRL1"
    idf.saveas('/Users/15142/Desktop/IBPS/BGT/BGT.idf')
    
    fnames=[]
    for i in range (1,3):
        fname1 = "/Users/15142/Desktop/IBPS/BGT/BGT.idf"
        epwfile = "/Users/15142/Desktop/total-3OCCs-8.9/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw"
        idf = IDF(fname1,epwfile)
        idf.saveas('/Users/15142/Desktop/IBPS/BGT/BGT%d.idf'%(i))
        #globals()["idfname{}".format(i)] = 
        fnames.append('/Users/15142/Desktop/IBPS/BGT/BGT%d.idf'%(i))
        #files = os.listdir(path)
        #fnames = [f for f in files if f[-4:] == '.idf']
    from eppy.modeleditor import IDF
    from eppy.runner.run_functions import runIDFs
    idfs = (IDF(fname, epwfile) for fname in fnames)
    runs = ((idf, make_options(idf) ) for idf in idfs)
    num_CPUs = 4
    runIDFs(runs, num_CPUs)
    
    TCENERGY=[]
    THENERGY=[]
    TEUI=[]
    TINC=[]
    TDCR=[]
   
    for i in range (1,3):
        Data=pd.read_csv('BGT%d.csv'%(i))
        
        CENERGY=Data['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Energy [J](TimeStep)'].sum()*2.78*10**(-7)
        HENERGY=Data['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Heating Energy [J](TimeStep)'].sum()*2.78*10**(-7)
        INC=Data['EMS:SP_Incoutput [](TimeStep)'].sum()
        DCR=Data['EMS:SP_Dcroutput [](TimeStep)'].sum()
        TCENERGY.append(CENERGY)
        THENERGY.append(HENERGY)
        TINC.append(INC)
        TDCR.append(DCR)
        path = '/Users/15142/Desktop/IBPS/BGT/'
        file = 'BGT%dTable.csv'%(i)
        f = open(path+file,'rt')
        reader = csv.reader(f)
        csv_list = []
        for l in reader:
            csv_list.append(l)
        f.close()
        df = pd.DataFrame(csv_list)
        EUI=df.iloc[14,3]
        TEUI.append(EUI)
    # change type of EUI array from string to float 
    TEUI = np.array(TEUI, dtype=np.float32)
    TEUI=TEUI*0.278
    
    print (np.average(TCENERGY))
    print (np.average(THENERGY))
    print (np.average(TINC))  
    print (np.average(TDCR))
    
    return np.sum(TEUI)


# import time
# starttime = time.time()
# if __name__ == '__main__':
    
    
#    main()
# print('Time taken = {} seconds'.format(time.time() - starttime))
    


# In[8]:


algorithm_param = {'max_num_iteration': 8,'population_size':4,'mutation_probability':0.35,'elit_ratio': 0.05,'crossover_probability': 0.7,'parents_portion': 0.3,'crossover_type':'uniform','max_iteration_without_improv':None}


# In[9]:


varbound= np.array([[0,16],[0,10],[0,10],[0,10],[0,8],[0,6],[0,3],[0,3],[0,10],[0,7],[0,4],[0,7]])
#varbound=np.array[np.arange(1,4,1),np.arange(0,20,5)]
#varbound=np.array[A,B]
vartype=np.array([['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int']])
#np.array([[1,5]])

model=ga(function=main,dimension=12,variable_type_mixed=vartype,variable_boundaries=varbound, function_timeout=20000,
         algorithm_parameters=algorithm_param)
model.run()


# In[ ]:


fname1 = "/Users/15142/Desktop/total-3OCCs-8.9/M1Final3OCCSOV.idf"
epwfile = "/Users/15142/Desktop/total-3OCCs-8.9/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw"
idf = IDF(fname1,epwfile)


# In[ ]:


re=np.array([18424.35566875, 18424.35566875, 18180.50782155, 18180.50782155, 18180.50782155, 18148.73733398, 18148.73733398, 18148.73733398, 18148.73733398, 18148.73733398, 18148.73733398, 18148.73733398, 18148.73733398, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462, 17907.3528462])


# In[ ]:


plt.plot(re)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


fname1 = "/Users/15142/Desktop/total-3OCCs-8.9/M1Final3OCCSOV.idf"
epwfile = "/Users/15142/Desktop/total-3OCCs-8.9/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw"
idf = IDF(fname1,epwfile)


# In[ ]:


idf.run(expandobjects= True, readvars= True)


# In[ ]:


# code for reading the csv file for EUI
import csv
import pandas as pd
TEUI=[]
for i in range (1,2),
    path = '/Users/15142/Desktop/total-3OCCs-8.9/'
    file = 'M%dFinal3OCCSOVTable.csv'%(i)
    f = open(path+file,'rt')
    reader = csv.reader(f)
    csv_list = []
    for l in reader:
        csv_list.append(l)
    f.close()
df = pd.DataFrame(csv_list)
EUI=df.iloc[14,3]
TEUI.append(EUI)


# In[ ]:


import csv
import pandas as pd
path = '/Users/15142/Desktop/total-3OCCs-8.9/'
file = 'eplustbl.csv'
f = open(path+file,'rt')
reader = csv.reader(f)


# In[ ]:


csv_list = []
for l in reader:
    csv_list.append(l)
f.close()


# In[ ]:


csv_list


# In[ ]:


reader


# In[ ]:


df = pd.DataFrame(csv_list)


# In[ ]:


EUI=df.iloc[14,3]


# In[ ]:


TEUI.append(EUI)


# In[ ]:


df


# In[ ]:


df.iloc[14,3]


# In[ ]:


df.head(20)


# In[ ]:


maxh=np.arange(22,30.1,0.5)


# In[ ]:


import numpy as np


# In[ ]:


maxh


# In[ ]:


minhandc=np.arange(15,20.1,0.5)


# In[ ]:


minhandc


# In[ ]:


maxc=np.arange(25,30.1,0.5)


# In[ ]:


maxc


# In[ ]:


u=np.arange(1.4,2.2,0.1)


# In[ ]:


u


# In[ ]:


SHGC=np.arange(0.3,0.61,0.05)


# In[ ]:


SHGC


# In[ ]:


Blind=np.arange(0.05,0.2,0.05)


# In[ ]:


Blind


# In[ ]:


q=np.arange(0.2,0.71,0.05)


# In[ ]:


q


# In[ ]:


r=np.arange(0.55,1.26,0.1)


# In[ ]:


r


# In[ ]:


s=np.arange(0.13,0.54,0.1)


# In[ ]:


s


# In[ ]:


w=np.arange(0.4,1.2,0.1)


# In[ ]:


w


# In[ ]:


fname1 = "/Users/15142/Desktop/IBPS/BGT/BGT.idf"
epwfile = "/Users/15142/Desktop/total-3OCCs-8.9/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw"
idf = IDF(fname1,epwfile)


# In[ ]:


Windowmaterial = idf.idfobjects['WindowMaterial:SimpleGlazingSystem'][0]
Windowmaterial.UFactor=1.5


# In[ ]:


print(Windowmaterial.UFactor)


# In[ ]:


idf.saveas('/Users/15142/Desktop/IBPS/BGT/BGT.idf')


# In[ ]:


y=[0,90,180,270]


# In[ ]:


g=y[2]


# In[ ]:


g


# In[ ]:


x=[1,5]


# In[ ]:


x


# In[ ]:


x=[1,5]


# In[ ]:


x


# In[ ]:


x[0]


# In[ ]:


x[1]


# In[ ]:


x=(0,16)


# In[ ]:


x


# In[ ]:


x[0]


# In[ ]:


x[1]


# In[ ]:


x=[1, 16]


# In[ ]:


x[0]


# In[ ]:


x[1]


# In[ ]:


x[2]


# In[ ]:




