import numpy as np
import scipy as sp
import pandas as pd
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.mps import TransferMatrix
from tenpy.networks.mpo import MPO
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import tebd
from scipy.special import ellipe
from tenpy.algorithms import dmrg
import tenpy.linalg.np_conserved as npc
import os
import pickle
import copy
import logging

from itdvp import *



def make_eps(v, ramp='linear'):
    if ramp == 'smooth':
        def eps(t):
            return v*t - (4.0/27.0)*((v*t)**3)
    else:
        def eps(t):
            return v*t
    return eps

def make_gt(v,eps):
    def gt(t):
        return 1 - eps(t)
    return gt

def make_Jt(v,eps,path):
    if path == "both":
        def Jt(t):
            return 1 + eps(t)
    else:
        def Jt(t):
            return 0*t + 1
    return Jt

def exact_energy(J, g):
    if not J==0:
        return -J*(2.0/np.pi)*abs(1.0+(g/J))*ellipe(4.0*(g/J)/((1.0+(g/J))**2))
    else:
        return -g

class TEBD_Evolver:
    
    first = True
    def __init__(self, options):
        self.options = options
        
    def evolve(self, psi, M):
        eng = tebd.TEBDEngine(psi, M, self.options)
        eng.run()
        return psi

class TDVP_Evolver:
    
    first = True
    def __init__(self, options, options_tebd, chi):
        self.options = options
        self.options_tebd = options_tebd
        self.chi = chi
        
    def evolve(self, psi, M):
        if np.mean(psi.chi) == self.chi:
            if self.first:
                print('swapping to TDVP')
                psi = make_uniform(psi)
                self.first = False
            eng = iTDVPEngine(psi, M, self.options)
        else:
            eng = tebd.TEBDEngine(psi, M, self.options_tebd)
        eng.run()
        return psi
        
class Repository:
    
    def __init__(self, options):
        self.options = options
        self._set_io_paths()
        
    def _set_io_paths(self):
        self.data_dir = self.options.get('data_dir','.data/')
        self.results_dir = self.options.get('res_dir','.results/')
        self.log_dir = self.options.get('log_dir','.log/')
        
class Repository_Pickle(Repository):
    
    def __init__(self, options):
        super().__init__(options)
        
    def write(self, file, df):
        df.to_csv(self.results_dir + file, index=False)
    
    def save(self, file, data):
        with open(self.data_dir + file, 'wb') as f:
            for k,v in data.items():
                pickle.dump(v, f)
                
    def log(self, file, df):
        df.to_csv(self.log_dir + file, index=False)
        
    def load(self, file, data):
        if not os.path.exists(self.data_dir + file):
            return False
        try:
            with open(self.data_dir + file, 'rb') as f:
                for key in data:
                    data[key] = pickle.load(f)
            return True
        except:
            return False

class uMPS(MPS):

    def __init__(self, sites, Bs, SVs, bc='finite', form='B', norm=1.):
        super().__init__(sites, Bs, SVs, bc=bc, form=form, norm=norm)
        self.B_dict = {}
        self.S = []
        
    def get_B(self, i, form='B', copy=False, cutoff=1E-16, label_p=None):
        if not isinstance(form, str):
            if form in list(self._valid_forms.values()):
                form = list(self._valid_forms.keys())[list(self._valid_forms.values()).index(form)]
        if not form in self.B_dict:
            B = super().get_B(i, form=form, copy=copy, cutoff=cutoff)
        else:
            B = self.B_dict[form]
        if label_p is not None:
            B = self._replace_p_label(B, label_p)
        return B
    
    def set_B(self, i, B, form='B'):
        for j in range(self.L):
            super().set_B(j, B)
        self.B_dict[form] = B
        
    def set_SL(self, i, s):
        for j in range(self.L):
            super().set_SL(j, s)
        self.S = s
        
    def set_SR(self, i, s):
        for j in range(self.L):
            super().set_SR(j, s)
        self.S = s
    
    def get_S(self):
        return self.get_SL(0)
    
    def get_SL(self, i):
        if len(self.S) == 0:
            print('S not stored, creating.')
            S = super().get_SL(i)
            self.S = S
        return self.S
    
    def get_SR(self, i):
        if len(self.S) == 0:
            print('S not stored, creating.')
            S = super().get_SR(i)
            self.S = S
        return self.S
        
    def set_B_dict(self, Bs):
        for k,v in Bs.items():
            for j in range(self.L):
                super().set_B(j, v, form=k)
            self.B_dict[k] = v
            
    def copy(self):
        res = super().copy()
        res.B_dict = self.B_dict
        return res
    
    
def make_uMPS(psi_in):
    return uMPS(psi_in.sites, psi_in._B, psi_in._S, bc='infinite', form='B', norm=1.0)

def make_uniform(psi):
    _chis = psi.chi

    #RCF
    TB = TransferMatrix(psi, psi, shift_bra=1, shift_ket=0)
    es_B, evs_B = TB.eigenvectors()
    U = evs_B[0].split_legs(['(vL.vL*)']).iset_leg_labels(['vL','vR'])
    _B = psi.get_B(1)
    B = npc.tensordot(_B, U, axes=('vR','vL'))
    B = (B/npc.norm(B))*np.sqrt(_chis[1])

    #LCF
    TA = TransferMatrix(psi, psi, shift_bra=1, shift_ket=0, form='A', transpose=True)
    es_A, evs_A = TA.eigenvectors()
    V = evs_A[0].split_legs(['(vR*.vR)'])
    Vdag = V.conj()
    _A = psi.get_B(1, form='A')
    A = npc.tensordot(_A, Vdag, axes=('vR','vR*'))
    A = (A/npc.norm(A))*np.sqrt(_chis[1])

    s = 0.5*(psi.get_SL(0) + psi.get_SL(1))
    upsi = make_uMPS(psi)
    upsi.set_B(0, A, form='A')
    upsi.set_B(0, B)
    upsi.set_SL(0, s)
    return upsi


