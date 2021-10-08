import numpy as np
import scipy as sp
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.mps import TransferMatrix
from tenpy.networks.mpo import MPO
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import tebd
from scipy.special import ellipe
from tenpy.algorithms import dmrg
import tenpy.linalg.np_conserved as npc

import copy
import logging
import pandas as pd
from util import *
from itdvp import *


from collections import OrderedDict
from typing import Any

class KZM:
    
    _data = {}
    def __getattr__(self, attr: str) -> Any:
        if attr in self._data:
            return self._data.get(attr)
        return super().__getattr__(attr)

    def __setattr__(self, key: str, value: Any) -> None:
        if key not in dir(self):
            self._data[key] = value
            
    def __init__(self, options):
        
        self._data = {}
        self.options = options
        self._set_params()
        self._set_options()
        self._set_couplings()
        self._set_repo()
        self._set_evolver()
        
        if options.get('load', True):
            if not self.load_checkpoint(): # new job
                self.get_gs(self.gi, self.Ji)
                self.setup_evolution()
            
        else:
            self.calc_ground_state(self.gi, self.Ji)
            self.setup_evolution()
        # completely new simulation
        
    def calc_ground_state(self, g, J):
        model_params = dict(L=2, J=J, g=g, bc_MPS='infinite', conserve=None)
        M = TFIChain(model_params)
        product_state = ["up"] * M.lat.N_sites
        psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
        eng = dmrg.TwoSiteDMRGEngine(psi, M, self.dmrg_options)
        self.E0, self.psi0 = eng.run()
        self.save_gs(g, J)
            
    def setup_evolution(self):
        self.Es.append(self.E0)
        self.Es_exact.append(exact_energy(self.Ji,self.gi))
        self.gs.append(self.gi)
        self.Js.append(self.Ji)
        self.ts.append(-self.ti)
        self.Ss.append(self.Si)
        self.chis.append(np.mean(self.psi0.chi))
        self.psi = self.psi0.copy()
        self.psi0 = None
        self.E0 = None

        
    def evolve(self):
        
        for t, g, J in zip(self._ts, self._gs, self._Js):
            self.t = t
            self.g = g
            self.J = J
            model_params = dict(L=self.psi.L, J=J, g=g, bc_MPS='infinite', conserve=None)
            M = TFIChain(model_params)
            self.psi = self.Evolver.evolve(self.psi, M)
            self.Es.append(np.mean(M.bond_energies(self.psi)))
            self.Ss.append(self.psi.entanglement_entropy()[-1])
            self.Es_exact.append(exact_energy(J,g))
            self.ts.append(t)
            self.gs.append(g)
            self.Js.append(J)
            self.chis.append(np.mean(self.psi.chi))
            print('v:',self.v,'g:',round(g,5),'J:',round(J,5),'chi:',self.chis[-1],'E:',round(self.Es[-1],6),'E_exact:',round(self.Es_exact[-1],6))
            self.write_log()
            self.save_checkpoint()
        print('done')
        print('*'*80)
        self.measurement()
        self.write_results()
        
    def measurement(self):
        g = self.gc if (self.endpoint == 'critical') else 0.0
        J = 2.0 if ((self.path == 'both') and (self.endpoint == 'ising')) else 1.0
        self.get_gs(g, J)
        self.l = abs(self.psi0.overlap(self.psi))

    def write_results(self):
        df = pd.DataFrame({'v': [self.v], 
                           'chi': [self.chi], 
                           'E': [self.Es[-1]],
                           'E0': [self.E0], 
                           'l': [self.l], 
                           'S': [self.Ss[-1]], 
                           'dt': [self.dt], 
                           'path': [self.path],
                           'endpoint': [self.endpoint],
                           'ramp': [self.ramp],
                            'evolver': [self.evolver]})
        self.repo.write(self.hash_results(), df)
    
    def write_log(self):
        df = pd.DataFrame({'t': self.ts, 'g': self.gs, 'J': self.Js, 'chi': self.chis, 'E': self.Es, 'S': self.Ss, 'E_exact': self.Es_exact})
        df['ramp'] = self.ramp
        df['path'] = self.path
        df['chi_max'] = self.chi
        df['endpoint'] = self.endpoint
        df['v'] = self.v
        df['dt'] = self.dt
        df['evolver'] = self.evolver
        self.repo.log(self.hash_log(), df)
        
    def save_checkpoint(self):
        self.repo.save(self.hash_checkpoint(), self.dict_checkpoint())
    
    def save_gs(self, g, J):
        self.repo.save(self.hash_gs(g, J), self.dict_gs())
        
    def dict_checkpoint(self):
        res = OrderedDict(psi=self.psi,
                          Es=self.Es, 
                          Ss = self.Ss,
                          Es_exact=self.Es_exact,
                          gs=self.gs, 
                          Js=self.Js, 
                          ts=self.ts, 
                          chis=self.chis,
                          g=self.g,
                          J=self.J,
                          t=self.t,
                          tdvp_options=self.tdvp_options,
                          first=self.Evolver.first,
                          tebd_options=self.tebd_options)
        self.test_save = res.copy()
        return res
    
    def dict_gs(self):
        res = OrderedDict(psi0=self.psi0,
                          E0=self.E0)
        return res
    
    def hash_results(self):
        hash_dict = OrderedDict({
            'ramp' : self.ramp,
            'path' : self.path,
            'endpoint' : self.endpoint,
            'gi' : self.gi,
            'v' : self.v,
            'dt' : self.dt,
            'chi' : self.chi,
            'evolver' : self.evolver
        })
        return self._hash('results:', hash_dict)
    
    def hash_log(self):
        hash_dict = OrderedDict({
            'ramp' : self.ramp,
            'path' : self.path,
            'endpoint' : self.endpoint,
            'gi' : self.gi,
            'v' : self.v,
            'dt' : self.dt,
            'chi' : self.chi,
            'evolver' : self.evolver
        })
        return self._hash('log:', hash_dict)
    
    def hash_checkpoint(self):
        hash_dict = OrderedDict({
            'ramp' : self.ramp,
            'path' : self.path,
            'gi' : self.gi,
            'v' : self.v,
            'dt' : self.dt,
            'chi' : self.chi,
            'evolver' : self.evolver
        })
        return self._hash('checkpoint:', hash_dict)
    
    def hash_gs(self, g, J):
        hash_dict = OrderedDict({
            'g' : g,
            'J' : J,
            'chi' : self.chi
        })
        return self._hash('ground_state:', hash_dict)
    
    def get_gs(self, g, J):
        self.g = g
        self.J = J
        if not self.load_gs(g, J): # new ground state
            self.calc_ground_state(g, J)
        self.Si = (self.psi0.entanglement_entropy())[-1]
        
    def load_checkpoint(self):
        load_dict = self.dict_checkpoint()
        if self.repo.load(self.hash_checkpoint(), load_dict):
            self.test_load = copy.deepcopy(load_dict)
            self._data.update(load_dict)
            self._update_couplings()
            self._set_evolver()
            self.Evolver.first = self.first
            return True
        return False

    def _set_evolver(self):
        evolver = self.options.get('evolver', 'TDVP')
        if evolver == 'TDVP':
            self.Evolver = TDVP_Evolver(self.tdvp_options, self.tebd_options, self.chi)
        elif evolver == 'TEBD':
            self.Evolver = TEBD_Evolver(self.tebd_options)

    def load_gs(self, g, J):
        load_dict = self.dict_gs()
        if self.repo.load(self.hash_gs(g, J), load_dict):
            self._data.update(load_dict)
            return True
        return False
        
        
    def _set_repo(self):
        self.repo = Repository_Pickle(self.repo_options)
    
    def _update_couplings(self):
        It = np.argwhere(self._ts == self.t)[0][0]
        self._ts = self._ts[It+1:]
        self._gs = self._gs[It+1:]
        self._Js = self._Js[It+1:]
        
    def _set_couplings(self):
        path = self.path
        ramp = self.ramp
        endpoint = self.endpoint
        eps = make_eps(self.v, ramp)
        gt = make_gt(self.gc, eps)
        Jt = make_Jt(eps, path)
        self._set_times(path, ramp, endpoint)
        self._gs = gt(self._ts)
        self._Js = Jt(self._ts)
        
    def _set_times(self, path, ramp, endpoint):

        if ramp == "linear":
            tf = 1.0 / self.v
            if path == "both":
                self.ti = tf
                self.gi = 2.0
                self.Ji = 0.0
            else:
                self.ti = (self.gi-1) * (1.0 / self.v)
                self.Ji = 1.0
        elif ramp == "smooth":
            tf = 1.5 * (1.0 / self.v)
            self.ti = tf
            self.gi = 2.0
            if  path == "both":
                self.Ji = 0.0
            else:
                self.Ji = 1.0
        
        self._ts = np.array(list(np.arange(-self.ti + self.dt, 0, self.dt)) + [0])
        if not endpoint == "critical":
            self._ts = np.array(list(self._ts) + list(np.arange(self.dt, tf, self.dt)) + [tf])
        
            
    def _set_options(self):
        
        self.tdvp_options = {
                                'dt' : self.dt, 
                                'N_steps': 1,
                                'N_env': 5,
                                'lanczos_options': {
                                    'N_min' : 3,
                                    'N_max' : 40,
                                    'reortho' : True
                                },
        }
        
        self.tebd_options = {
                                'order': 2,
                                'dt' : self.dt,
                                'N_steps': 1,
                                'trunc_params': {
                                    'chi_max': self.chi,
                                    'svd_min': 1.e-10
                                },
        }
        
        self.dmrg_options = {'mixer': True, 
                             'trunc_params': { 'chi_max': self.chi, 'svd_min': 1.e-10 } , 
        }
        
        self.repo_options = {
            
        }
        
        self.tebd_options.update(self.options.get('tebd_options',{}))
        self.tdvp_options.update(self.options.get('tdvp_options',{}))
        self.repo_options.update(self.options.get('repo_options',{}))
        self.dmrg_options.update(self.options.get('dmrg_options',{}))
        
    def _set_params(self):
        
        self.chi = self.options.get('chi', 10)
        self.v = self.options.get('v', 0.1)
        self.gi = self.options.get('gi', 3.0)
        self.gc = self.options.get('gc', 1.0)
        self.dt = self.options.get('dt', 0.1)
        self.path = self.options.get('path', 'both')
        self.ramp = self.options.get('ramp', 'linear')
        self.endpoint = self.options.get('endpoint', 'critical')
        self.evolver = self.options.get('evolver', 'TDVP')
        
        # defaults that need to be set
        self.psi0 = None
        self.psi = None
        self.psif = None
        self.Si = None
        self.E0 = None
        self.Ef = None
        self.g = None
        self.J = None
        self.t = None
        
        self.Es = []
        self.Ss = []
        self.Es_exact = []
        self.gs = []
        self.Js = []
        self.ts = []
        self.chis = []

    def _hash(self, label, hash_dict):
        res = label
        for k,v in hash_dict.items():
            res += '-'+k+'='+str(v)
        return res





