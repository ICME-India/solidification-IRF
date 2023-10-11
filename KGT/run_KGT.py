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
from scipy.optimize import bisect
from scipy.special import expi

class KGT():
    
    def __init__(self,obj):
        self.co=np.array(obj['Alloy_composition'])
        self.solutes=obj['Solutes']
        self.num_solutes=len(self.solutes)
        self.ke=np.array(obj['Eqm_partition_coefficient'])
        self.me=np.array(obj['Eqm_liquidus_slope'])
        self.Tl=obj["Liquidus_temperature"]
        self.muk=obj['Kinetic_coeff']
        self.D=np.array(obj["Diffusion_coefficient"])
        self.Gamma=obj["Gibbs_Thomson_coefficient"]
        self.vdi=obj["Interface_diffusive_speed"]
        self.GL=obj["Thermal_gradient"]
        
    
    def Ivantsov(self,p):
        '''
        

        Parameters
        ----------
        p : numpy array
            Peclet numbers.

        Returns
        -------
        list
            Ivantsov solution.

        '''
        ans=[]
        for a in p:
            if a>699.0:
                z=1.0
            else:
                z=-a*np.exp(a)*expi(-a)
            ans.append(z)
        return np.array(ans)
    
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
    
    def get_clstar(self,kv,pc):
        '''
        

        Parameters
        ----------
        kv : numpy array
             kinetic partition coefficient.
        pc : numpy array
            Solutal Peclet Number.

        Returns
        -------
        numpy array
            Liquid composition at tip.

        '''
        term=(1-kv)*self.Ivantsov(pc)
        return np.divide(self.co,(1-term))
        
    
    def iterate(self,vmin,vstep,vmax,R1,R2):
        '''
        

        Parameters
        ----------
        vmin : float
            Minimum velocity at which iteration starts.
        vstep : float
            Stepping velocity
        vmax : TYPE
            Maximum velocity at which iteration ends.
        R1 : float
            Lower bound of guess for radius.
        R2 : float
            Upper bound of guess for radius.

        Returns
        -------
        data : list
            velocity,radius,temperature.

        '''
        data=[]
        self.v=vmin
        Rg1=R1
        Rg2=R2
        while self.v<vmax:
            R=bisect(self.get_radius,Rg1,Rg2)
            Rg1=R/10
            Rg2=R*10.0
            answer=self.get_data(R*1E-6)
            T=answer[0]
            print(self.v,R,T)
            args=(np.array([self.v,R*1E-6]),answer)
            data.append(np.concatenate(args))
            if self.v>1.0:
                vstep=1E-2
            self.v=self.v+vstep
        return data
    
    def eta_c(self,kv, pc):
        '''
        Solutal stability function

        Parameters
        ----------
        kv : numpy array
            kinetic partition coefficient.
        pc : numpy array
            solutal peclet number.

        Returns
        -------
        ans : numpy array
            

        '''
        buf1=2*kv
        buf2=(2*kv)-1+ np.sqrt(1+(2*np.pi*np.reciprocal(pc))**2)
        ans=1-np.divide(buf1,buf2)
        return ans
    
    def get_radius(self,Rg):
        '''
        Function to be passed to scipy solver

        Parameters
        ----------
        Rg : float
            Scaled radius.

        Returns
        -------
        eqn : float
            Equation=0.

        '''
        R=Rg*1E-6
        kv=(self.ke+(self.v/self.vdi))/(1+(self.v/self.vdi))
        mv=self.get_mv(kv)
        pc=self.v*R*0.5*np.reciprocal(self.D)
        clstar=self.get_clstar(kv,pc)
        eta=self.eta_c(kv,pc)
        eqn1= (4*np.pi*np.pi)*self.Gamma/(R*R)
        eqn2=np.sum(pc*mv*clstar*eta*(1-kv))
        eqn2=2*eqn2/R
        eqn= eqn1+eqn2+self.GL
        return eqn
    
    def get_data(self,R):
        '''
        

        Parameters
        ----------
        R : float
            tip radius.

        Returns
        -------
        ans : numpy array
            Temperature,kinetic partition coefficient,liq composition at tip.

        '''
        kv=(self.ke+(self.v/self.vdi))/(1+(self.v/self.vdi))
        mv=self.get_mv(kv)
        pc=self.v*R*0.5*np.reciprocal(self.D)
        clstar=self.get_clstar(kv,pc)
        buf1=np.sum(mv*clstar)
        buf2=np.sum(self.me*self.co)
        T= self.Tl+buf1-buf2-(2*self.Gamma/R)-(self.v/self.muk)-(self.GL*self.D[0]/self.v)
        args=(np.array([T]),kv,clstar)
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
Rguess1=obj["Rguess_min_microns"]
Rguess2=obj["Rguess_max_microns"]
fname=obj["Filename"]

args=sys.argv[1]
path=args[:-5]
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)

#Perform the calculation
print(15*"#")
print("KGT IRF Calculation starts")
print(15*"#")
ob=KGT(obj)
answer=ob.iterate(vmin,vstep,vmax,Rguess1,Rguess2)

#Write the output to csv in a nice format
print(15*"#")
print("Calculation Complete and  output file written at ",fname)
kvs=['kv_'+i for i in solutes]
clstars=['clstar_'+i for i in solutes]
vrt=['v','R','T']
l=[vrt,kvs,clstars]
cols=[item for sublist in l for item in sublist]
df=pd.DataFrame(np.row_stack(answer),columns=cols)
df.to_csv(path+"/"+fname,index=True)

#Plot for immediate visualization
print(15*"#")
print("Calculation Complete and  output image written at ",fname[:-3]+"jpg")
plt.plot(df['v'],df['T'])
plt.xscale('log')
plt.title("KGT Model")
plt.xlabel("Soldification velocity (m/s)")
plt.ylabel("Temperature (K)")
plt.savefig(path+"/"+fname[:-3]+"jpg")
plt.show()