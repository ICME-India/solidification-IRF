#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:09:12 2022

@author: hariharan
"""

from tc_python import *
from scipy.optimize import fsolve,root
from scipy.special import expi
import numpy as np
import pandas as pd

class IRF():
    
    def __init__(self,obj,eqm_obj):
        self.base_element= obj['Base_element']
        self.solutes=obj['Solutes']
        self.num_solutes=len(self.solutes)
        self.phase=obj['Solid_phase']
        self.co=np.array(obj['Alloy_composition'])
        self.Tl_eqm=obj['Liquidus_temperature']
        self.eqm_obj=eqm_obj
        self.v0=obj['Max_speed']
        self.vdb=obj['Bulk_diffusive_speed']
        self.vdi=obj['Interface_diffusive_speed']
        self.vm=obj['Molar_volume']
        self.DHF=obj["Enthalpy_of_fusion"]
        self.CP=obj["CP"]
        self.TQ=self.DHF/self.CP
        self.D=np.array(obj["Diffusivity"])
        self.Gamma=obj["GibbsThomson_coefficient"]
        self.DTS=obj["Solid_thermal_diffusivity"]
        self.DTL=obj["Liquid_thermal_diffusivity"]
        self.KS=obj["Solid_thermal_conductivity"]
        self.KL=obj["Liquid_thermal_conductivity"]
        self.GL=obj["Thermal_gradient"]
        
    def get_deltaG(self,C,T):
        '''
        Gibbs energy difference

        Parameters
        ----------
        C : numpy array
           composition of solid and liquid
        T : float
            temperature

        Returns
        -------
        dG : float
            

        '''
        cl=C[:self.num_solutes]
        cs=C[-self.num_solutes:]
        clf= np.append(cl,1-np.sum(cl))
        csf= np.append(cs,1-np.sum(cs))
        liquid_potentials= self.mu(cl,T,"liquid") 
        solid_potentials= self.mu(cs,T,self.phase)
        del_mu_elements=solid_potentials-liquid_potentials
        e2t2= np.sum(clf*del_mu_elements)
        if self.v < self.vdb:
            term1=np.sum(0.5*del_mu_elements*(self.v/self.vdb)*(self.v/self.vdb)*(csf-clf))
            dG= -term1+e2t2
        else:
            dG=e2t2
        
        return dG
        
   
    def C_equations(self,guess,T,pc):
        '''
        Function to obtain solid composition

        Parameters
        ----------
        guess : numpy array
           composition of solid and liquid
        T : float
            temperature
        pc : numpy array
            Peclet number

        Returns
        -------
        eqns : numpy array
            Eqn=0

        '''
        cl=guess[:self.num_solutes]
        cs=guess[-self.num_solutes:]
        liquid_potentials= self.mu(cl,T,"liquid") 
        solid_potentials= self.mu(cs,T,self.phase)
        del_mu_elements=solid_potentials-liquid_potentials     
        clf= np.append(cl,1-np.sum(cl))
        csf= np.append(cs,1-np.sum(cs))

        if self.v< self.vdb:            
            psi=1-(self.v/self.vdb)**2
            md= self.Md(cl,T)
            mdsum= np.sum(md)
            coeff1=np.sum(md*psi*del_mu_elements)
            Jlist=md*((del_mu_elements*psi)-(coeff1/mdsum))
        else:
            Jlist=np.zeros(len(self.solutes)+1)

        J=self.v*(clf-csf)/self.vm
       
        eqns=Jlist[:-1]-J[:-1]
        
        if self.v < self.vdb:
            k=np.divide(cs,cl)
            den=1-((1-k)*self.Ivantsov(pc))
            ans=cl-np.divide(self.co,den)
        else:
            ans=cl-self.co
            
        eqns=np.append(eqns,ans)
       
        
        return eqns

    def Md(self,cl,t):
        
        #Md term in Wang Model
        #chemical potential derivative
        cl_temp=cl
        dc=1e-5
        derivs=np.zeros(len(self.solutes)+1)
        for i in range(len(self.solutes)):
            cl_temp[i]=cl[i]-dc
            mu1= self.muil(cl_temp,t,self.solutes[i])
            cl_temp[i]=cl[i]+dc
            mu2= self.muil(cl_temp,t,self.solutes[i])
            derivs[i]=(mu2-mu1)/(2*dc)
            cl_temp=cl
        
        cl_temp= cl - (dc/len(self.solutes))
        mu1= self.muil(cl_temp,t,self.base_element)
        cl_temp= cl + (dc/len(self.solutes))
        mu2= self.muil(cl_temp,t,self.base_element)
        derivs[-1]=(mu2-mu1)/(2*dc)
        
        md= self.vdi*np.reciprocal(derivs)/self.vm
        
        return md        
    
    
    def muil(self,conc,t,ele):
        #Chemical potential in liquid
        temp_obj=self.eqm_obj
        cond=self.set_multiple_conditions(conc)
        mu_obj=(temp_obj
                .set_phase_to_suspended(self.phase)
                .set_phase_to_entered("Liquid")
                .set_condition("T",t)
                .run_poly_command(cond)
                .calculate())
        return mu_obj.get_value_of("mu("+ele+")")
    
    def  mu(self,conc,t,inc_phase):
        #Chemical potential of a phase
        temp_obj=self.eqm_obj
        cond=self.set_multiple_conditions(conc)
        chem_pot=np.zeros(len(self.solutes)+1)
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
        chem_pot[-1]=mu_obj.get_value_of("mu("+self.base_element+")")
        for i in range(len(self.solutes)):
            chem_pot[i]=mu_obj.get_value_of("mu("+self.solutes[i]+")")
        return chem_pot
    
    def set_multiple_conditions(self,conc,comp_choice="mole"):
        if comp_choice=='mass':
            c="W("
        else:
            c="X("
        temp=["s-c"]
        for i in range(len(self.solutes)):
            temp.append(c+self.solutes[i]+")="+str(conc[i]))
        return (' '.join(temp))
    
    def get_term2(self,R,T,cl,cs):
        '''
        Get solutal term in stability expression

        Parameters
        ----------
        R : float
            tip radius
        T : float
            temperature
        cl : float
            liquid composition
        cs : float
           solid composition

        Returns
        -------
        float
            

        '''
        k=np.divide(cs,cl)
        omega2=(2*np.pi/R)**2.0
        omegaC=self.termC+np.sqrt((self.termC*self.termC)+(omega2/self.psi))
        factor=self.B*omegaC
        dervsC=self.get_C_derivatives(R,T,cl,cs)
        dervsT=self.get_T_derivatives(R,T,cl,cs)
        Mnum=-dervsC[0]
        Mden=dervsT[0]+(8.314*self.v/self.v0)
        M=np.divide(Mnum,Mden)
        Nnum=factor+(self.v*(k-1))+(self.v*cl*dervsC[1])
        den=self.v*cl*dervsT[1]
        N=np.divide(Nnum,den)
        Gil=(cs-cl)*2*self.termC
        zetaNum=Gil*(self.v-factor)
        zeta=np.divide(zetaNum,den)
        ratio=np.divide(M,N)
        termNum=np.sum(ratio*zeta)
        termDen=1+np.sum(ratio)
        term=np.divide(termNum,termDen)
        return term        
        
    
    def get_term1(self,R):
        '''
        First term in stability expression
        thermal and curvature term

        Parameters
        ----------
        R : float
            radius

        Returns
        -------
        ans : float
            

        '''
        omega2=(2*np.pi/R)**2.0
        sigma=1/(4*np.pi*np.pi)
        pt=self.v*R*0.5/self.DTL
        term=1/(sigma*pt*pt)
        etaS=1+(1/np.sqrt(1+term))
        etaL=1-(1/np.sqrt(1+term))
        #GS=self.GL+(self.TQ*self.v/(self.DTL))
        GS=self.GL
        ans=(0.5*self.GL*etaL)+(0.5*GS*etaS)+(self.Gamma*omega2)
        return ans
    
    def get_C_derivatives(self,R,T,cl,cs):
        '''
        Get derivatives with respect to composition

        Parameters
        ----------
        R : float
            tip radius
        T : float
            temperature
        cl : float
            liquid composition
        cs : float
           solid composition

        Returns
        -------
        list
            Gibbs energy derivative and partition coefficient derivative

        '''
        v_original=self.v
        delta_v=self.v/10.0
        guess=np.append(cl,cs)
        #1st iter
        self.v=v_original+delta_v
        pc1=self.v*R*0.5*np.reciprocal(self.D)
        X1=fsolve(self.C_equations,x0=guess,args=(T,pc1),maxfev=1000)
        dg1=self.get_deltaG(X1,T)
        #2nd iter
        self.v=v_original-delta_v
        pc2=self.v*R*0.5*np.reciprocal(self.D)
        X2=fsolve(self.C_equations,x0=guess,args=(T,pc2),maxfev=1000)
        dg2=self.get_deltaG(X2,T)
        cl1=X1[:self.num_solutes]
        cs1=X1[-self.num_solutes:]
        k1=np.divide(cs1,cl1)
        cl2=X2[:self.num_solutes]
        cs2=X2[-self.num_solutes:]
        k2=np.divide(cs2,cl2)
        #print(dg1,dg2,cl1,cl2)
        ddG=(dg1-dg2)*np.reciprocal(cl1-cl2)
        ddK=np.divide((k1-k2),(cl1-cl2))
        self.v=v_original
        return [ddG,ddK]
    
    def get_T_derivatives(self,R,T,cl,cs):
        '''
        Get derivatives with respect to temperature

        Parameters
        ----------
        R : float
            tip radius
        T : float
            temperature
        cl : float
            liquid composition
        cs : float
           solid composition

        Returns
        -------
        list
            Gibbs energy derivative and partition coefficient derivative

        '''
        delta_T=0.01
        guess=np.append(cl,cs)
        pc=self.v*R*0.5*np.reciprocal(self.D)
        #1st iter
        T1=T+delta_T
        X1=fsolve(self.C_equations,x0=guess,args=(T1,pc),maxfev=1000)
        dg1=self.get_deltaG(X1,T1)
        #2nd iter
        T2=T-delta_T
        X2=fsolve(self.C_equations,x0=guess,args=(T2,pc),maxfev=1000)
        dg2=self.get_deltaG(X2,T2)
        cl1=X1[:self.num_solutes]
        cs1=X1[-self.num_solutes:]
        k1=np.divide(cs1,cl1)
        cl2=X2[:self.num_solutes]
        cs2=X2[-self.num_solutes:]
        k2=np.divide(cs2,cl2)
        ddG=(dg1-dg2)/(2*delta_T)
        ddK=(k1-k2)/(2*delta_T)
        return [ddG,ddK]
        

        
    def Ivantsov(self,p):
        '''
        Ivantsov solution

        Parameters
        ----------
        p : numpy array
            Peclet number

        Returns
        -------
        numpy array
           

        '''
        ans=[]
        for x in p:
            if x>699.0:
                z=1.0
            else:
                z = -x*np.exp(x)*expi(-x)
            ans.append(z)
        return np.array(ans)
        
        
    def RT_calc(self,guessRT):
        '''
        Function used as argument for solver

        Parameters
        ----------
        guessRT : numpy array
            Scaled radius and temperature.

        Returns
        -------
        list
           Eqns=0

        '''
        R=guessRT[0]*1E-6
        Ti=guessRT[1]*self.Tl_eqm
        dTR=2*self.Gamma/R
        T=Ti+dTR
        pc=self.v*R*0.5*np.reciprocal(self.D)
        X=fsolve(self.C_equations,x0=self.guess_cl_cs,args=(T,pc),maxfev=1000)
        cl=X[:len(self.solutes)]
        cs=X[-len(self.solutes):]
        self.guess_cl_cs=X
        term1=self.get_term1(R)
        term2=self.get_term2(R,T,cl,cs)
        outputR= -term1-term2
        dg=self.get_deltaG(X,T)
        outputT=T+(self.v0*dg/(8.314*self.v))
        #print(R,Ti,cl,cs,term1,term2,dg)
        
        return [outputR,outputT]
    
    def iter_coupled(self,vmin,vstep,vmax,guessT,guess0,Rgg):
        '''
        Main function

        Parameters
        ----------
        vmin : float
            Minimum velocity from which iteration starts.
        vstep : float
            Step velocity.
        vmax : float
            Maximum velocity at which iteration ends.
        guessT : float
            Scaled guess temperature.
        guess0 : numpy array
            Liquid and solid compositions
        Rgg : float
            Scaled guess radius

        Returns
        -------
        data : list
            velocity,radius,temperature,liq and sol composition

        '''
        data=[]
        self.v=vmin
        self.guess_cl_cs=guess0
        Tg=guessT
        guess=np.append(Rgg,Tg)
        while self.v<vmax:
            self.guess_cl_cs=guess0
            self.psi=1-((self.v/self.vdb)**2)
            self.B=self.psi*self.D
            self.termC=np.divide(self.v,(2*self.B))
            try:
                answer=root(self.RT_calc,x0=guess)
                guess=answer.x
                args=(np.array([self.v]),guess,self.guess_cl_cs)
                data.append(np.concatenate(args))
                print(data[-1])
                vstep=min(2*10**(np.floor(np.log10(self.v))),1e-2)
                if self.v>3.0:
                    vstep=0.1
                self.v=self.v+vstep
            except Exception as e:
                print("Error occured in",self.v)
                print(e)
                return data
        return data
        
        
        
        

   