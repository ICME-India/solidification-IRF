# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:52:49 2023

@author: hariharan
"""

import json
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


class Linear_planar():
    
    def __init__(self,obj):
        self.co=np.array(obj['Alloy_composition'])
        self.solutes=obj['Solutes']
        self.num_solutes=len(self.solutes)
        self.ke=np.array(obj['Eqm_partition_coefficient'])
        self.me=np.array(obj['Eqm_liquidus_slope'])
        self.Tm=obj["Solvent_melting_point"]
        self.muk=obj['Kinetic_coeff']
        self.vdi=obj["Interface_diffusive_speed"]
        
    
    def get_mv(self,kv):
        '''
        

        Parameters
        ----------
        kv : numpy array
            kinetic partition coefficient.

        Returns
        -------
        numpy array
            kinetic liquidus slope.

        '''
        prefactor=np.divide(self.me,(1-self.ke))
        num=1-(kv*(1-np.log(np.divide(kv,self.ke))))        
        return prefactor*num
      
    
    def iterate(self,vmin,vstep,vmax):
        '''
        

        Parameters
        ----------
        vmin : float
            Minimum velocity at which iteration starts.
        vstep : float
            Velocity steps for iteration.
        vmax : float
            Maximum velocity at which iteration stops.

        Returns
        -------
        data : list
            velocity,temperature.

        '''
        data=[]
        self.v=vmin
        while self.v<vmax:
            answer=self.get_data()
            T=answer[0]
            print(self.v,T)
            args=(np.array([self.v]),answer)
            data.append(np.concatenate(args))
            if self.v>1.0:
                vstep=1E-2
            self.v=self.v+vstep
        return data
    
    
    def get_data(self):
        '''
        

        Returns
        -------
        ans : list
            Temperature,kinetic partition coefficient.

        '''
        kv=(self.ke+(self.v/self.vdi))/(1+(self.v/self.vdi))
        mv=self.get_mv(kv)
        buf1=np.sum(mv*self.co)
        T= self.Tm+buf1-(self.v/self.muk)
        args=(np.array([T]),kv)
        ans=np.concatenate(args)
        return ans
    
#Parse json file
with open(sys.argv[1], 'r') as myfile:
    data=myfile.read()

obj = json.loads(data)
solutes=obj["Solutes"]
vmin=obj["Minimum_velocity"]
vstep=obj["Step_velocity"]
vmax=obj["Maximum_velocity"]
fname=obj["Filename"]

args=sys.argv[1]
path=args[:-5]
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)

#Perform the calculation
print(15*"#")
print("Linear Planar IRF Calculation starts")
print(15*"#")
ob=Linear_planar(obj)
answer=ob.iterate(vmin,vstep,vmax)

#Write the output to csv in a nice format
print(15*"#")
print("Calculation Complete and  output file written at ",fname)
kvs=['kv_'+i for i in solutes]
vrt=['v','T']
l=[vrt,kvs]
cols=[item for sublist in l for item in sublist]
df=pd.DataFrame(np.row_stack(answer),columns=cols)
df.to_csv(path+"/"+fname,index=True)

#Plot for immediate visualization
print(15*"#")
print("Calculation Complete and  output image written at ",fname[:-3]+"jpg")
plt.plot(df['v'],df['T'])
plt.xscale('log')
plt.title("Linear Planar Model")
plt.xlabel("Soldification velocity (m/s)")
plt.ylabel("Temperature (K)")
plt.savefig(path+"/"+fname[:-3]+"jpg")
plt.show()