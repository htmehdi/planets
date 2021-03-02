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

from eppy.pytest_helpers import do_integration_tests
from eppy.runner.run_functions import install_paths, EnergyPlusRunError
from eppy.runner.run_functions import multirunner
from eppy.runner.run_functions import run
from eppy.runner.run_functions import runIDFs

Ncores=int(sys.argv[1])


# In[2]:


path = "/speed-scratch/z_khoras"


# In[3]:


iddfile = path+ '/EP-8-9/EnergyPlus-8-9-0/Energy+.idd'
IDF.setiddname(iddfile)


# In[4]:


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
    iddfile = path+ '/EP-8-9/EnergyPlus-8-9-0/Energy+.idd'
    IDF.setiddname(iddfile)
    epwfile = path+'/Final8Var3BG/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'

    #mapping for maximum amount for threshold
    a1=np.arange(300,701,20)
    a2 = a1[int(X[0])]*0.0929
    # mapping for minimum amount for threshold
    b1=np.arange(0,301,20)
    b2= b1[int(X[1])]*0.0929
    #mapping for window VT
    c1=np.arange(0.3,0.81,0.05)
    c2=c1[int(X[2])]
    #mapping visible reflectance for floor
    d1=np.arange(0,1.1,0.1)
    d2=d1[int(X[3])]
    #mapping visible reflectance for roof
    e1=d1[int(X[4])]
    #mapping visible reflectance for walls
    f1=np.arange(0,1.1,0.1)
    f2=f1[int(X[5])]
    # mapping axis to an array
    g1=[0,90,180,270]
    g2=g1[int(X[6])]
    #mapping for blind visible transmittance
    h1=np.arange(0.05,0.21,0.05)
    h2=h1[int(X[7])]
    #mapping wwr
    i1=np.arange(0.2,0.71,0.05)
    i2=i1[int(X[8])]
    #w = np.asscalar(X)
    fname1 = path +'/Final8Var3BG/BGM.idf'
    epwfile = path+'/Final8Var3BG/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'

    idf = IDF(fname1,epwfile)
    #the second variable
    Gunayocclightingmodel = idf.idfobjects['EnergyManagementsystem:program'][6]
    Gunayocclightingmodel.Program_Line_24 ="IF Light_SP <" + str(b2) 
    Gunayocclightingmodel.Program_Line_25 =" set Light_SP =" + str(b2)
    # the first variable:
    Gunayocclightingmodel.Program_Line_26 ="ELSEIF Light_SP>" + str(a2)
    Gunayocclightingmodel.Program_Line_27 ="set Light_SP =" + str(a2)
    # the third variable:
    Windowmaterial = idf.idfobjects['WindowMaterial:SimpleGlazingSystem'][0]
    Windowmaterial.Visible_Transmittance=c2
    #the forth variable (floor visible absorptance)
    F16_acoustic_tile_floor=idf.idfobjects['Material'][13]
    F16_acoustic_tile_floor.Visible_Absorptance=d2
    #the fifth variable (roof visible absorptance)
    F16_acoustic_tile_roof=idf.idfobjects['Material'][5]
    F16_acoustic_tile_roof.Visible_Absorptance=e1
    #wall visible absorptance
    G01a_19mm_gypsum_board=idf.idfobjects['Material'][2]
    G01a_19mm_gypsum_board.Visible_Absorptance=f2
    #the north axes
    office=idf.idfobjects['Building'][0]
    office.North_Axis=g2
    #blind visible Transmittance
    Blind_material=idf.idfobjects['WindowMaterial:Shade'][0]
    Blind_material.Visible_Transmittance=h2
    idf.saveas(path +'/Final8Var3BG/BGN.idf')
    #WWr
    from geomeppy import IDF
    
    fname2 = path +'/Final8Var3BG/BGM.idf'
    idf1 = IDF(fname2,epwfile)
    idf1.set_wwr(wwr=0, wwr_map={180: i2}, force=True, construction= "Exterior Window")
    idf1.saveas(path +'/Final8Var3BG/BGM.idf')
    #setting wshCTRL
    from eppy.modeleditor import IDF
    fname1 = path +'/Final8Var3BG/BGM.idf'
    epwfile = path+'/Final8Var3BG/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
    idf = IDF(fname1,epwfile)
    sub_surface = idf.idfobjects['FenestrationSurface:Detailed'][0]
    sub_surface.Shading_Control_Name="wshCTRL1"
    idf.saveas(path +'/Final8Var3BG/BGM.idf')
    
    fnames=[]
    for i in range (1,33):
        fname1 = path +'/Final8Var3BG/BGM.idf'
        epwfile = path+'/Final8Var3BG/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
        idf = IDF(fname1,epwfile)
        idf.saveas(path +'/Final8Var3BG/BGM%d.idf'%(i))
        
        fnames.append(path +'/Final8Var3BG/BGM%d.idf'%(i))
        
    from eppy.modeleditor import IDF
    from eppy.runner.run_functions import runIDFs
    idfs = (IDF(fname, epwfile) for fname in fnames)
    runs = ((idf, make_options(idf) ) for idf in idfs)
    num_CPUs = Ncores
    runIDFs(runs, num_CPUs)
    
    TRELC=[]
    TEUI=[]
    TON=[]
    TOFF=[]
   
    for i in range (1,33):
        Data=pd.read_csv(path +'/Final8Var3BG/BGM%d.csv'%(i))
        ELC=Data['LIGHT:Lights Electric Energy [J](TimeStep)'].sum()*2.78*10**(-7)
        ON=Data['EMS:switchonoutput [](TimeStep)'].sum()
        OFF=Data['EMS:switchoffoutput [](TimeStep)'].sum()
        TRELC.append(ELC)
        TON.append(ON)
        TOFF.append(OFF)
        
        f = open(path +'/Final8Var3BG/BGM%d.idf'%(i))
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
    
    print (np.average(TRELC))
    print (np.average(TON))  
    print (np.average(TOFF))
    
    return np.sum(TEUI)




algorithm_param = {'max_num_iteration': 50,'population_size':15,'mutation_probability':0.35,'elit_ratio': 0.05,'crossover_probability': 0.7,'parents_portion': 0.3,'crossover_type':'uniform','max_iteration_without_improv':None}





varbound= np.array([[0,20],[0,15],[0,10],[0,10],[0,10],[0,10],[0,3],[0,3],[0,10]])

vartype=np.array([['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int']])


model=ga(function=main,dimension=9,variable_type_mixed=vartype,variable_boundaries=varbound, function_timeout=20000,
         algorithm_parameters=algorithm_param)
model.run()

