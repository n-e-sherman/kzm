import numpy as np
import pandas as pd
import random
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import tebd
from collections import OrderedDict

from util import *
from state import *
from solver import *

class GroundState(State):
    
    def __init__(self, options):
        super().__init__(options)
        self._set_params()
        self._set_solver()
        
        if not (self.options.get('load', True) and (self.load_gs())):
            self.calc_gs()
                
    def measure(self):
        res_dict = measure_state(self.psi0, self.M)
        res_dict.update(self.measurement_tags())
        self.measurements = pd.DataFrame(res_dict)
        self.repo.write('gs/' + self.hash_measurement(), self.measurements)
        
    def calc_gs(self):
        psi = self.get_initial()
        self.E0, self.psi0 = self.Solver.solve(self.model_params, psi)
        self.save_gs()
    
    def save_gs(self):
        self.repo.save(self.hash_gs(), self.dict_gs())
        
    def load_gs(self):
        load_dict = self.dict_gs()
        if self.repo.load(self.hash_gs(), load_dict):
            self._data.update(load_dict)
            return True
        return False
    
    def dict_gs(self):
        res = OrderedDict(psi0=self.psi0,
                          E0=self.E0,
                          dmrg_options=self.dmrg_options,
                          tebd_options=self.tebd_options)
        return res
    
    def hash_gs(self):
        hash_dict = OrderedDict({
            'g' : self.g,
            'J' : self.J,
            'chi' : self.chi,
            'solver': self.solver,
            'conserve': str(self.conserve)
        })
        return self._hash('ground_state:', hash_dict)
    
    def hash_measurement(self):
        hash_dict = OrderedDict({
            'g' : self.g,
            'J' : self.J,
            'chi' : self.chi,
            'solver': self.solver,
            'conserve': str(self.conserve)
        })
        return self._hash('measurement:', hash_dict)
    
    def _hash(self, label, hash_dict):
        res = label
        for k,v in hash_dict.items():
            res += '-'+k+'='+str(v)
        return res
    
    def get_initial(self):

        if self.conserve == 'parity':
            product_state = ["up"] * self.M.lat.N_sites
        elif self.conserve is None:
            site_tensor = np.array([random.uniform(0,1),random.uniform(0,1)])
            site_tensor = site_tensor / np.linalg.norm(site_tensor)
            product_state = [site_tensor] * self.M.lat.N_sites
        else:
            raise ValueError("self.conserve given an unknown value: " + str(self.conserve))
        return MPS.from_product_state(self.M.lat.mps_sites(), product_state, bc=self.M.lat.bc_MPS)

    def measurement_tags(self):
        tags = {}
        tags['E_exact'] = exact_energy(self.J, self.g)
        tags['g'] = self.g
        tags['J'] = self.J
        tags['chi'] = self.chi
        tags['solver'] = self.solver
        tags['conserve'] = self.conserve
        return tags
        
    def _set_solver(self):
        solver = self.solver
        if solver == 'TEBD':
            self.Solver = TEBD_Solver(self.tebd_options)
        elif solver == 'DMRG1-bias':
            self.Solver = DMRG1_bias_Solver(self.dmrg_options)
        elif solver == 'DMRG1':
            self.Solver = DMRG1_Solver(self.dmrg_options)
        elif solver == 'DMRG2-bias':
            self.Solver = DMRG2_bias_Solver(self.dmrg_options)
        else:
            self.Solver = DMRG2_Solver(self.dmrg_options)
            
    def _set_params(self):
        
        self.chi = self.options.get('chi', 10)
        self.g = self.options.get('g', 1.0)
        self.J = self.options.get('J', 1.0)
        self.solver = self.options.get('solver', 'DMRG2')
        self.conserve = self.options.get('conserve', None)
        self.model_params = dict(L=2, J=self.J, g=self.g, bc_MPS='infinite', conserve=self.conserve)
        self.M = TFIChain(self.model_params)
        
        # defaults that need to be set
        self.psi0 = None
        self.E0 = None
        