#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:09:12 2022

@author: hariharan
"""

from tc_python import *
from scipy.optimize import fsolve
import math
import numpy as np
import pandas as pd

class IRF():
    
    def __init__(self,obj,eqm_obj):
        self.base_element= obj['Base_element']
        self.solutes=obj['Solutes']
        self.phase=obj['Solid_phase']
        self.cl=np.array(obj["Liquid_composition"])
        self.tl=obj['Liquidus_temperature']
        self.eqm_obj=eqm_obj
        self.v0=obj['Maximum_speed']
        self.vdi=obj['Interface_diffusive_speed']
        self.eqm_obj=eqm_obj
        self.drag=obj["Solute_drag"]
        self.sum_solutes_liquid=np.sum(self.cl)
        
    def equations(self,guess):
        '''
        Functions which has all the equations

        Parameters
        ----------
        guess : numpy array
            solid composition and scaled temperature.

        Returns
        -------
        eqns : numpy array
            Equation=0

        '''
        cs=guess[:-1]
        T=guess[-1]*self.tl
        const=8.314*math.log(1-(self.v/self.v0))
        liquid_potentials= self.mu(self.cl,T,"liquid") # 0-Al, 1-Si
        solid_potentials= self.mu(cs,T,self.phase)
        del_mu_elements=[]
    
        for i in range(len(liquid_potentials)):
            del_mu_elements.append(solid_potentials[i]-liquid_potentials[i])
        
        sum_solutes_solid=0
        for j in cs:
            sum_solutes_solid=sum_solutes_solid+ j
        
        if self.drag==0:
            sum_solutes=sum_solutes_solid
            compo=cs
        else:
            sum_solutes=self.sum_solutes_liquid
            compo=self.cl
        delG=(1-sum_solutes)*del_mu_elements[0]
        for k in range(1,len(del_mu_elements)):
            delG= delG+ (compo[k-1]*del_mu_elements[k])
            
        del_mu_elementsD=[del_mu_elements[0]-(8.314*T*math.log((1-sum_solutes_solid)/(1-self.sum_solutes_liquid)))]
        for l in range(1,len(del_mu_elements)):
            del_mu_elementsD.append(del_mu_elements[l]- 8.314*T*math.log(cs[l-1]/self.cl[l-1]))
        
        kappa=[[0 for dummy1 in range(len(self.solutes)+1)] for dummy2 in range(len(self.solutes)+1)]
        for x in range(1+len(self.solutes)):
            for y in range(1,len(self.solutes)+1):
                kappa[x][y]=math.exp(-(del_mu_elementsD[y]-del_mu_elementsD[x])/(8.314*T))
                
        eqns=[]
        
        for m in range(len(self.solutes)):
            temp0=(self.cl[m]-cs[m])*(self.v/self.vdi)
            temp1= (1-self.sum_solutes_liquid)*cs[m]- kappa[0][m+1]*(1-sum_solutes_solid)*self.cl[m]
            for n in range(len(self.solutes)):
                if m==n:
                    continue
                else:
                    temp1=temp1+(self.cl[n]*cs[m])-(kappa[n+1][m+1]*cs[n]*self.cl[m])
                    if kappa[n+1][m+1]==0:
                        print("Warning kappa zero")
            eqns.append(temp1-temp0)
        eqns.append(delG - (T*const))
        return eqns
    
    def  mu(self,conc,t,inc_phase):
        '''
        Calculates chemical potential

        Parameters
        ----------
        conc : numpy array
            composition of a phase.
        t : float
            temperature.
        inc_phase : string
            phase

        Returns
        -------
        chem_pot : list
            chemical potential.

        '''
        temp_obj=self.eqm_obj
        cond=self.set_multiple_conditions(conc)
        if inc_phase=="liquid":
            sus_phase=self.phase
        else:
            sus_phase="liquid"
        
        mu_obj=(temp_obj
                .set_phase_to_suspended(sus_phase)
                .set_phase_to_entered(inc_phase)
                .set_condition("T",t)
                .run_poly_command(cond)
                .calculate())
        chem_pot=[mu_obj.get_value_of("mu("+self.base_element+")")]
        for i in self.solutes:
            chem_pot.append(mu_obj.get_value_of("mu("+i+")"))
        return chem_pot
    
    def set_multiple_conditions(self,conc,comp_choice="mole"):
        '''
        Function to effectively set conditions for eqm calculations

        Parameters
        ----------
        conc : numpy array
            composition.
        comp_choice : string, optional
             The default is "mole". Can also be mass but do not use

        Returns
        -------
        string
            string to set condition

        '''
        if comp_choice=='mass':
            c="W("
        else:
            c="X("
        temp=["s-c"]
        for i in range(len(self.solutes)):
            temp.append(c+self.solutes[i]+")="+str(conc[i]))
        return (' '.join(temp))
    
    def iter_vel(self,vmin,vmax,vstep,guess0):
        '''
        Main Function to iter

        Parameters
        ----------
        vmin : float
            Minimum velocity at which iteration starts.
        vmax : float
            Maximum velocity at which iteration stops.
        vstep : float
            Stepping velocity
        guess0 : numpy array
            Solid composition and scaled temperature

        Returns
        -------
        data : list
            velocity,solid composition, temperature

        '''
        data=[]
        self.v=vmin
        guess=guess0
        while self.v<vmax:
            self.v=self.v+vstep
            try:
                ans= fsolve(self.equations,x0=guess)
            except Exception as e:
                print(e)
                print("An Exception Occured - Printing interim file")
                df=pd.DataFrame(np.row_stack(data))
                df.to_csv('interim_result.csv',index=False)
                return data
            data.append(np.concatenate(([self.v],ans)))
            guess=ans
        return data
    

    