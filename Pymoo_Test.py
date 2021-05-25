#!/usr/bin/env python
# coding: utf-8
# %%
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
import pymoo
import autograd.numpy as anp

from pymoo.problems.util import load_pareto_front_from_file
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
#Ncores=int(sys.argv[1])


# %%
#path = "/speed-scratch/z_khoras/"


# %%
iddfile = path+ '/EP-8-9/EnergyPlus-8-9-0/Energy+.idd'
IDF.setiddname(iddfile)


# %%
"""multiprocessing runs"""

# using generators instead of a list
# when you are running a 100 files you have to use generators

import os 
from eppy.modeleditor import IDF
from eppy.runner.run_functions import runIDFs

class MultiNGBT(Problem):
    
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=30, xu=35, elementwise_evaluation=True, type_var=anp.double)
        #self.args = args
    
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


    def _evaluate(self, X, out,*args, **kwargs):
        
        from eppy.modeleditor import IDF
        iddfile = path+ '/EP-8-9/EnergyPlus-8-9-0/Energy+.idd'
        IDF.setiddname(iddfile)
        epwfile = path+'/SenBTNT_New/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
        #mapping for heating maximum threshold 
        #maxh=np.arange(22,30.1,0.5)
        a = X[0]


        # mapping for heating minimum threshold
        #minhandc=np.arange(15,20.1,0.5)
        b= 18

        #mapping for cooling maximum threshold
        #maxc=np.arange(25,30.1,0.5)
        c=X[1]

        #mapping for colling minimum threshold
        d=18

        #mapping for U-factor glazing
        #e1=np.arange(1.4,2.2,0.1)
        e2=1.4

        #mapping for SHGC
        #SHGC=np.arange(0.3,0.61,0.05)
        f=0.5499

        # mapping axis to an array
        #g1=[0,90,180,270]
        g2=0

        #mapping for blind solar transmittance
        #Blind=np.arange(0.05,0.2,0.05)
        h=0.1

        #mapping wwr
        #i1=np.arange(0.2,0.71,0.05)
        i2=0.2

        #mapping for roof R-value
        #j1=np.arange(0.55,1.26,0.1)
        j2=1.0499

        #mapping for floor R-value
        #k1=np.arange(0.13,0.54,0.1)
        k2=0.33

        #mapping for exterior wall R-value
        #l1=np.arange(0.2,0.6,0.05)
        l2=0.449


        #Mapping for period
        m = 29

        # mapping for luxmean
        #n1=np.arange(0,301,20)
        n2= 140

        #mapping for window VT
        #o1=np.arange(0.3,0.81,0.05)
        o2=0.749

        #mapping visible reflectance for floor
        #p1=np.arange(0,1.1,0.1)
        p2=0.7

        #mapping visible reflectance for roof
        q1=0

        #mapping visible reflectance for walls
        #r1=np.arange(0,1.1,0.1)
        r2=0

        #mapping for blind visible transmittance
        #s1=np.arange(0.05,0.21,0.05)
        s2=0.2

        # for LR in thermostat model
        # LR = np.arange(0.0005, 0.01, 0.0005)
        #LR = round(X,5)
        LR = 0.008

        #for time interval(TI)

        #TIS = "Minute==30 || Minute==60"
        #"""
        if X==1:
            TIS= "Minute==10 || Minute==20 || Minute==30 || Minute==40 || Minute==50 || Minute==60"
        elif X==2:
            TIS= "Minute==20 || Minute==40 || Minute==60"
        elif X==3:
            TIS= "Minute==30 || Minute==60"
        elif X==4:
            TIS= "Minute==60"
        elif X==5:
            TIS= "CurrentTime==1.5 || CurrentTime==3.0 || CurrentTime==4.5 || CurrentTime==6.0 || CurrentTime==7.5 ||CurrentTime==9.0 || CurrentTime==10.5 || CurrentTime==12.0 || CurrentTime==13.5 || CurrentTime==15.0 || CurrentTime==16.5 || CurrentTime==18.0 || CurrentTime==19.5 || CurrentTime==21.0 || CurrentTime==22.5 || CurrentTime==24.0"
        elif X==6:
            TIS= "CurrentTime==0.0 || CurrentTime==2.0 || CurrentTime==4.0 || CurrentTime==6.0 || CurrentTime==8.0 || CurrentTime==10.0 || CurrentTime==12.0 || CurrentTime==14.0 || CurrentTime==16.0 || CurrentTime==18.0 || CurrentTime==20.0 || CurrentTime==22.0 || CurrentTime==24.0"
        elif X==7:
            TIS= "CurrentTime==2.5 || CurrentTime==5.0 || CurrentTime==7.5 || CurrentTime==10.0 || CurrentTime==12.5 ||CurrentTime==15.0 || CurrentTime==17.5 || CurrentTime==20.0 || CurrentTime==22.5 || CurrentTime==24.0"
        elif X==8:
            TIS= "CurrentTime==0.0 || CurrentTime==3.0 || CurrentTime==6.0 || CurrentTime==9.0 || CurrentTime==12.0 || CurrentTime==15.0 || CurrentTime==18.0 || CurrentTime==21.0 || CurrentTime==24.0"
        elif X==9:
            TIS= "CurrentTime==3.5 || CurrentTime==7 || CurrentTime==10.5 || CurrentTime==14 || CurrentTime==17.5 ||CurrentTime==21.0 || CurrentTime==24.0"
        else:
            TIS= "CurrentTime==0.0 || CurrentTime==4.0 || CurrentTime==8.0 || CurrentTime==12.0 || CurrentTime==16.0 || CurrentTime==20.0 || CurrentTime==24.0"

        #"""  

        fname1 = path +'/PMBTNG/BT_NG_T_Tr.idf'
        epwfile = path+'/PMBTNG/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'

        idf = IDF(fname1,epwfile)
        occthermostatmodel = idf.idfobjects['EnergyManagementsystem:program'][8]
        occthermostatmodel.Program_Line_74 ="IF HSP < " + str(b) #maybe we can use directly "set x=" + str(X[0])
        occthermostatmodel.Program_Line_75 ="set HSP = " + str(b)
        # the second variable:
        occthermostatmodel.Program_Line_76="ELSEIF HSP >" + str(a)
        occthermostatmodel.Program_Line_77="set HSP = " + str(a)
        # the third variable:
        occthermostatmodel.Program_Line_80 ="IF CSP < " + str(d)
        occthermostatmodel.Program_Line_81 ="set CSP = " + str(d)
        # the forth variable
        occthermostatmodel.Program_Line_82 ="ELSEIF CSP > " + str(c)
        occthermostatmodel.Program_Line_83 ="set CSP = " + str(c)

        # LR for termostat model
        occthermostatmodel.Program_Line_9 ="set LR = " + str(LR)

        #TI for thermostat model
        occthermostatmodel.Program_Line_53 = "If " + TIS



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
        occlightingmodel = idf.idfobjects['EnergyManagementsystem:program'][4]
        occlightingmodel.Program_Line_7 ="set x=" + str(m) #maybe we can use directly "set x=" + str(X[0])
        # the second variable:
        occlightingmodel.Program_Line_3="set luxmean=" + str(n2)
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

        idf.saveas(path +'/PMBTNG/BT_NG_T_Tr.idf')

        #WWr
        from geomeppy import IDF

        fname2 = path +'/PMBTNG/BT_NG_T_Tr.idf'
        idf1 = IDF(fname2,epwfile)
        idf1.set_wwr(wwr=0, wwr_map={180: i2}, force=True, construction= "Exterior Window")
        idf1.saveas(path +'/PMBTNG/BT_NG_T_Tr.idf')
        #setting wshCTRL
        from eppy.modeleditor import IDF
        fname1 = path +'/PMBTNG/BT_NG_T_Tr.idf'
        epwfile = path+'/PMBTNG/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
        idf = IDF(fname1,epwfile)
        sub_surface = idf.idfobjects['FenestrationSurface:Detailed'][0]
        sub_surface.Shading_Control_Name="wshCTRL1"
        idf.saveas(path +'/PMBTNG/BT_NG_T_Tr.idf')
        fnames=[]
        
        for i in range (1,33):
            fname1 = path +'/PMBTNG/BT_NG_T_Tr.idf'
            epwfile = path+'/PMBTNG/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
            idf = IDF(fname1,epwfile)
            idf.saveas(path +'/PMBTNG/BT_NG_T_Tr%d.idf'%(i))

            fnames.append(path +'/PMBTNG/BT_NG_T_Tr%d.idf'%(i))

        from eppy.modeleditor import IDF
        from eppy.runner.run_functions import runIDFs
        idfs = (IDF(fname, epwfile) for fname in fnames)
        runs = ((idf, make_options(idf) ) for idf in idfs)
        num_CPUs = Ncores
        runIDFs(runs, num_CPUs)

        TC=[]
        TH=[]
        TEUI=[]
        TINC=[]
        TDCR=[]
        TL=[]
        TON=[]
        TOFF=[]

        for i in range (1,33):
            Data=pd.read_csv(path +'/PMBTNG/BT_NG_T_Tr%d.csv'%(i))

            #CENERGY=Data['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Energy [J](TimeStep)'].sum()*2.78*10**(-7)
            #HENERGY=Data['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Heating Energy [J](TimeStep)'].sum()*2.78*10**(-7)
            INC=Data['EMS:SP_Incoutput [](TimeStep)'].sum()
            DCR=Data['EMS:SP_Dcroutput [](TimeStep)'].sum()
            ELC=Data['LIGHT:Lights Electric Energy [J](TimeStep)'].sum()*2.78*10**(-7)
            ON=Data['EMS:countonoutput [](TimeStep)'].iloc[-1]
            OFF=Data['EMS:countoffoutput [](TimeStep)'].iloc[-1]
            #TRELC.append(ELC)
            TON.append(ON)
            TOFF.append(OFF)
            #TCENERGY.append(CENERGY)
            #THENERGY.append(HENERGY)
            TINC.append(INC)
            TDCR.append(DCR)

            file = path +'/PMBTNG/BT_NG_T_Tr%dTable.csv'%(i)
            f = open(file,'rt')
            reader = csv.reader(f)
            csv_list = []
            for l in reader:
                csv_list.append(l)
            f.close()
            df = pd.DataFrame(csv_list)
            EUI=df.iloc[14,3]
            Heating=df.iloc[49,6]
            Cooling=df.iloc[50,5]
            Lighting=df.iloc[51,2]
            TEUI.append(EUI)
            TH.append(Heating)
            TC.append(Cooling)
            TL.append(Lighting)
        # change type of EUI array from string to float 
        TEUI = np.array(TEUI, dtype=np.float32)
        TH = np.array(TH, dtype=np.float32)
        TC = np.array(TC, dtype=np.float32)
        TL = np.array(TL, dtype=np.float32)
        TEUI=TEUI*0.278 # MJ TO kWh
        TH=TH*278 #GJ TO kWh
        TC=TC*278 #GJ TO kWh
        TL=TL*278 #GJ TO kWh



        print (np.average(TH))
        print (np.average(TC))
        print (np.average(TINC))  
        print (np.average(TDCR))
        print (np.average(TL))
        print (np.average(TON))  
        print (np.average(TOFF))
        print (np.average(TEUI))
        obj1 = np.average(TEUI)
        obj2 = np.average(TOFF)
        
        out["F"] = anp.column_stack([obj1, obj2])
        #return (np.average(TEUI),np.average(TOFF))

    def _calc_pareto_front(self, *args, **kwargs):
        return load_pareto_front_from_file("MultiNGBT.pf")
    
#vectorized_problem = MultiNGBT()

# %%
problem = MultiNGBT()

algorithm = NSGA2(pop_size=5)

res = minimize(problem,
               algorithm,
               ("n_gen", 5),
               verbose=True,
               seed=5)

plot = Scatter()
plot.add(res.F, color="red")
plot.show()

# %%