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
    epwfile = path+'/Final8Var2BGT/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
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
    l1=np.arange(0.2,0.6,0.05)
    l2=l1[int(X[11])]
    #Mapping for max threshold
    m1 = np.arange(300,701,20)
    m2 = m1[int(X[12])]*0.0929
    # mapping for minimum threshold
    n1=np.arange(0,301,20)
    n2= n1[int(X[13])]
    #mapping for window VT
    o1=np.arange(0.3,0.81,0.05)
    o2=o1[int(X[14])]
    #mapping visible reflectance for floor
    p1=np.arange(0,1.1,0.1)
    p2=p1[int(X[15])]
    #mapping visible reflectance for roof
    q1=p1[int(X[16])]
    #mapping visible reflectance for walls
    r1=np.arange(0,1.1,0.1)
    r2=r1[int(X[17])]
    #mapping for blind visible transmittance
    s1=np.arange(0.05,0.21,0.05)
    s2=s1[int(X[18])]
    
    fname1 = path +'/Final8Var2BGT/BT_BG_S.idf'
    epwfile = path+'/Final8Var2BGT/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'

    idf = IDF(fname1,epwfile)
    occthermostatmodel = idf.idfobjects['EnergyManagementsystem:program'][8]
    occthermostatmodel.Program_Line_73 ="IF HSP < " + str(b) #maybe we can use directly "set x=" + str(X[0])
    occthermostatmodel.Program_Line_74 ="set HSP = " + str(b)
    # the second variable:
    occthermostatmodel.Program_Line_75="ELSEIF HSP >" + str(a)
    occthermostatmodel.Program_Line_76="set HSP = " + str(a)
    # the third variable:
    occthermostatmodel.Program_Line_79 ="IF CSP < " + str(d)
    occthermostatmodel.Program_Line_80 ="set CSP = " + str(d)
    # the forth variable
    occthermostatmodel.Program_Line_81 ="ELSEIF CSP > " + str(c)
    occthermostatmodel.Program_Line_82 ="set CSP = " + str(c)
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
    
    #Nagy OCC
    Gunayocclightingmodel = idf.idfobjects['EnergyManagementsystem:program'][6]
    Gunayocclightingmodel.Program_Line_24 ="IF Light_SP <" + str(n2) 
    Gunayocclightingmodel.Program_Line_25 =" set Light_SP =" + str(n2)
    # the first variable:
    Gunayocclightingmodel.Program_Line_26 ="ELSEIF Light_SP>" + str(m2)
    Gunayocclightingmodel.Program_Line_27 ="set Light_SP =" + str(m2)
    # the third variable:
    Windowmaterial = idf.idfobjects['WindowMaterial:SimpleGlazingSystem'][0]
    Windowmaterial.Visible_Transmittance=o2
    #the forth variable (floor visible absorptance)
    F16_acoustic_tile_floor=idf.idfobjects['Material'][13]
    F16_acoustic_tile_floor.Visible_Absorptance=p2
    #the fifth variable (roof visible absorptance)
    F16_acoustic_tile_roof=idf.idfobjects['Material'][5]
    F16_acoustic_tile_roof.Visible_Absorptance=q1
    #wall visible absorptance
    G01a_19mm_gypsum_board=idf.idfobjects['Material'][2]
    G01a_19mm_gypsum_board.Visible_Absorptance=r2
    #the north axes
    #blind visible Transmittance
    Blind_material=idf.idfobjects['WindowMaterial:Shade'][0]
    Blind_material.Visible_Transmittance=s2
    idf.saveas(path +'/Final8Var2BGT/BT_BG_S.idf')
  
    #WWr
    from geomeppy import IDF
    
    fname2 = path +'/Final8Var2BGT/BT_BG_S.idf'
    idf1 = IDF(fname2,epwfile)
    idf1.set_wwr(wwr=0, wwr_map={180: i2}, force=True, construction= "Exterior Window")
    idf1.saveas(path +'/Final8Var2BGT/BT_BG_S.idf')

    #setting wshCTRL
    from eppy.modeleditor import IDF
    fname1 = path +'/Final8Var2BGT/BT_BG_S.idf'
    epwfile = path+'/Final8Var2BGT/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
    idf = IDF(fname1,epwfile)
    sub_surface = idf.idfobjects['FenestrationSurface:Detailed'][0]
    sub_surface.Shading_Control_Name="wshCTRL1"
    idf.saveas(path +'/Final8Var2BGT/BT_BG_S.idf')
    
    fnames=[]
    for i in range (1,33):
        
        fname1 = path +'/Final8Var2BGT/BT_BG_S.idf'
        epwfile = path+'/Final8Var2BGT/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
        idf = IDF(fname1,epwfile)
        idf.saveas(path +'/Final8Var2BGT/BT_BG_S%d.idf'%(i))

        fnames.append(path +'/Final8Var2BGT/BT_BG_S%d.idf'%(i))

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
    TRELC=[]
    TON=[]
    TOFF=[]
   
    for i in range (1,33):
        Data=pd.read_csv(path +'/Final8Var2BGT/BT_BG_S%d.csv'%(i))
        
        CENERGY=Data['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Energy [J](TimeStep)'].sum()*2.78*10**(-7)
        HENERGY=Data['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Heating Energy [J](TimeStep)'].sum()*2.78*10**(-7)
        INC=Data['EMS:SP_Incoutput [](TimeStep)'].sum()
        DCR=Data['EMS:SP_Dcroutput [](TimeStep)'].sum()
        ELC=Data['LIGHT:Lights Electric Energy [J](TimeStep)'].sum()*2.78*10**(-7)
        ON=Data['EMS:countonoutput [](TimeStep)'].iloc[-1]
        OFF=Data['EMS:countoffoutput [](TimeStep)'].iloc[-1]
        TRELC.append(ELC)
        TON.append(ON)
        TOFF.append(OFF)
        TCENERGY.append(CENERGY)
        THENERGY.append(HENERGY)
        TINC.append(INC)
        TDCR.append(DCR)
        
        file = path +'/Final8Var2BGT/BT_BG_S%dTable.csv'%(i)
        f = open(file,'rt')
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
    print (np.average(TRELC))
    print (np.average(TON))  
    print (np.average(TOFF))
    
    return np.sum(TEUI)


# In[5]:


algorithm_param = {'max_num_iteration': 50,'population_size':19,'mutation_probability':0.35,'elit_ratio': 0.05,'crossover_probability': 0.7,'parents_portion': 0.3,'crossover_type':'uniform','max_iteration_without_improv':None}


# In[ ]:


varbound= np.array([[0,16],[0,10],[0,10],[0,10],[0,8],[0,6],[0,3],[0,3],[0,10],[0,7],[0,4],[0,7],[0,20],[0,15],[0,10],[0,10],[0,10],[0,10],[0,3]])

vartype=np.array([['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int'], ['int'],['int'],['int'],['int'],['int'],['int'],['int']])


model=ga(function=main,dimension=19,variable_type_mixed=vartype,variable_boundaries=varbound, function_timeout=20000,
         algorithm_parameters=algorithm_param)
model.run()

