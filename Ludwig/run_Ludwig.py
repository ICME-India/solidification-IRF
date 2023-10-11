#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:30:48 2022

@author: hariharan
"""
import json
from tc_python import *
import pandas as pd
import numpy as np
import os.path
import sys
import Ludwig_IRF
import matplotlib.pyplot as plt

def set_multiple_conditions():
    c="X("
    temp=["s-c"]
    for i in range(len(solutes)):
        temp.append(c+solutes[i]+")="+str(cl[i]))
    return (' '.join(temp))

#Read json file for reading parameters

with open(sys.argv[1], 'r') as myfile:
    data=myfile.read()

# parse file
obj = json.loads(data)

database=obj['Database']
base_element=obj['Base_element']
solutes=obj['Solutes']
cl=obj['Liquid_composition']
solid_phase=obj['Solid_phase']
cs=obj['Solid_composition']
tl=obj['Liquidus_temperature']
drag=obj['Solute_drag']
fname=obj['Filename']

vmin=obj['Minimum_velocity']
vmax=obj['Maximum_velocity']
vstep=obj['Step_velocity']

args=sys.argv[1]
path=args[:-5]
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)



if drag==0:
    print("No Solute Drag")
else:
    print("Solute Drag Active")


cols=['v']
for s in solutes:
    cols.append('Cs_'+s)
cols.append('T')

print(15*"#")
print("Ludwig IRF Calculation starts")
print(15*"#")

with TCPython() as start:
    eq_calculation= (
        start
            .set_cache_folder(os.path.basename(fname[:-4]) + "_cache")
            .select_database_and_elements(database, [base_element]+solutes)
            .without_default_phases()
            .select_phase("LIQUID")
            .select_phase(solid_phase)
            .get_system()
            .with_single_equilibrium_calculation()
            .disable_global_minimization()
            .run_poly_command(set_multiple_conditions())
            .set_condition("T",tl)
    )

    guess0=cs+[1]
    obj= Ludwig_IRF.IRF(obj,eq_calculation)
    
    ans=obj.iter_vel(vmin,vmax,vstep,guess0)
    df=pd.DataFrame(np.row_stack(ans),columns=cols)
    df['T']=df['T']*tl
    print(15*"#")
    print("Calculation Complete and  output file written at ",fname)
    df.to_csv(path+"/"+fname,index=False)
    
    #Plot for immediate visualization
    print(15*"#")
    print("Calculation Complete and  output image written at ",fname[:-3]+"jpg")
    plt.plot(df['v'],df['T'])
    plt.xscale('log')
    plt.title("Ludwig Model")
    plt.xlabel("Soldification velocity (m/s)")
    plt.ylabel("Temperature (K)")
    plt.savefig(path+"/"+fname[:-3]+"jpg")
    plt.show()