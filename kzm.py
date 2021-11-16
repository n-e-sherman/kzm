import numpy as np
import pandas as pd
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import tebd
from collections import OrderedDict

from util import *
from state import *
from evolver import *

class KZM(State):
                
    def __init__(self, options):
        super().__init__(options)
        self._set_params()
        self._set_couplings()
        self._set_evolver()
        
        if not (self.options.get('load', True) and (self.load_checkpoint())):
            self.get_initial_state(self.gi)
            self.setup_evolution()

# Main work
    def evolve(self):
        for t, g, J in zip(self._ts, self._gs, self._Js):
            self.t = t
            self.g = g
            self.J = J
            model_params = dict(L=2, J=J, g=g, bc_MPS='infinite', conserve=None)
            M = TFIChain(model_params)
            self.M = M
            self.psi = self.Evolver.evolve(self.psi, M)
            self.Es.append(np.mean(M.bond_energies(self.psi)))
            self.Ss.append(self.psi.entanglement_entropy()[-1])
            self.Es_exact.append(exact_energy(J,g))
            self.ts.append(t)
            self.gs.append(g)
            self.Js.append(J)
            self.chis.append(np.mean(self.psi.chi))
            self.Sxs.append(np.mean(self.psi.expectation_value('Sx')))
            self.Sys.append(np.mean(self.psi.expectation_value('Sy')))
            self.Szs.append(np.mean(self.psi.expectation_value('Sz')))
        
            print('v:',self.v,'g:',round(g,5),'J:',round(J,5), 'chi:', self.chi, 'Sx:', round(self.Sxs[-1],6), 'Sy:', round(self.Sys[-1],6), 'Sz:', round(self.Szs[-1],6))
            self.write_log()
            self.save_checkpoint()
        self.write_log()
        self.save_checkpoint()
        print('done')
        print('*'*80)
        
    def measure(self):
        res_dict = measure_state(self.psi, self.M)
        res_dict.update(self.measurement_tags())
        self.measurements = pd.DataFrame(res_dict)
        self.repo.write('kzm/' + self.hash_measurement(), self.measurements)

    def setup_evolution(self):
        self.Es.append(self.E0)
        self.Es_exact.append(exact_energy(self.Ji,self.gi))
        self.gs.append(self.gi)
        self.Js.append(self.Ji)
        self.ts.append(-self.ti)
        self.Ss.append(self.Si)
        self.Sxs.append(self.Sxi)
        self.Sys.append(self.Syi)
        self.Szs.append(self.Szi)
        self.chis.append(np.mean(self.psi0.chi))
        self.psi = self.psi0.copy()
        self.psi0 = None
        self.E0 = None
        
    def get_initial_state(self, g):
        self.g = g
        self.J = 0.0
        model_params = dict(L=2, J=self.J, g=self.g, bc_MPS='infinite', conserve=None)
        M = TFIChain(model_params)
        product_state = ["up"] * M.lat.N_sites
        psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
        self.psi0 = psi.copy()
        self.Si = np.mean((self.psi0.entanglement_entropy()))
        self.Sxi = np.mean(self.psi0.expectation_value('Sx'))
        self.Syi = np.mean(self.psi0.expectation_value('Sy'))
        self.Szi = np.mean(self.psi0.expectation_value('Sz'))

# writing results
    def write_log(self):
        df = pd.DataFrame({'t': self.ts, 'g': self.gs, 'J': self.Js, 
                         'chi': self.chis, 'E': self.Es, 'S': self.Ss, 
                         'Sx': self.Sxs, 'Sy': self.Sys, 'Sz': self.Szs, 
                         'E_exact': self.Es_exact})
        df['ramp'] = self.ramp
        df['path'] = self.path
        df['chi_max'] = self.chi
        df['endpoint'] = self.endpoint
        df['v'] = self.v
        df['dt'] = self.dt
        df['evolver'] = self.evolver
        self.repo.log(self.hash_log(), df)
        
# Saving data
    def save_checkpoint(self):
        self.repo.save(self.hash_checkpoint(), self.dict_checkpoint())
    
# Loading data
    def load_checkpoint(self):
        load_dict = self.dict_checkpoint()
        if self.repo.load(self.hash_checkpoint(), load_dict):
            self._data.update(load_dict)
            self._update_couplings()
            self._set_evolver()
            self.Evolver.first = self.first
            return True
        return False

    def _update_couplings(self):
        inds = np.argwhere(self._gs < self.g) # there are more g values to run, we assume g is decreasing throughout the run
        if len(inds) > 0:
            it = inds[0][0]
            self._ts = self._ts[it:]
            self._gs = self._gs[it:]
            self._Js = self._Js[it:]
        if self.g == self._gs[-1]: # job finished:
            self._ts = np.array([])
            self._gs = np.array([])
            self._Js = np.array([])
        model_params = dict(L=self.psi.L, J=self.J, g=self.g, bc_MPS='infinite', conserve=None)
        self.M = TFIChain(model_params)

# dictionary for saving / loading
    def dict_checkpoint(self):
        res = OrderedDict(psi=self.psi,
                          Es=self.Es, 
                          Ss = self.Ss,
                          Sxs = self.Sxs,
                          Sys = self.Sys,
                          Szs = self.Szs,
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
        return res

# Hashes    
    def hash_log(self):
        hash_dict = OrderedDict({
            'ramp' : self.ramp,
            'path' : self.path,
            'endpoint' : self.endpoint,
            'gi' : self.gi,
            'gc' : self.gc,
            'v' : self.v,
            'dt' : self.dt,
            'chi' : self.chi,
            'evolver' : self.evolver
        })
        return super()._hash('log:', hash_dict)
    
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
        return super()._hash('checkpoint:', hash_dict)
    
    def hash_measurement(self):
        hash_dict = OrderedDict({
            'ramp' : self.ramp,
            'path' : self.path,
            'endpoint' : self.endpoint,
            'gi' : self.gi,
            'gc' : self.gc,
            'v' : self.v,
            'dt' : self.dt,
            'chi' : self.chi,
            'evolver' : self.evolver
        })
        return self._hash('measurement:', hash_dict)
    
    def hash_kzm(self):
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
        return self._hash('kzm:', hash_dict)

# Set default values from options
    def measurement_tags(self):
        tags = {}
        tags['E_exact'] = exact_energy(self.J, self.g)
        tags['ramp'] = self.ramp
        tags['path'] = self.path
        tags['endpoint'] = self.endpoint
        tags['dt'] = self.dt
        tags['evolver'] = self.evolver
        tags['chi'] = self.chi
        tags['v'] = self.v
        tags['gc'] = self.gc
        return tags

    def _set_evolver(self):
        evolver = self.options.get('evolver', 'TDVP')
        if evolver == 'TDVP':
            self.Evolver = TDVP_Evolver(self.tdvp_options, self.tebd_options, self.chi)
        elif evolver == 'TEBD':
            self.Evolver = TEBD_Evolver(self.tebd_options)
        
    def _set_couplings(self):
        path = self.path
        ramp = self.ramp
        endpoint = self.endpoint
        eps = make_eps(self.v, ramp)
        self.gt = make_gt(self.gc, eps)
        self.Jt = make_Jt(eps, path)
        self._set_times(path, ramp, endpoint)
        self._gs = self.gt(self._ts)
        self._Js = self.Jt(self._ts)
        
    def _set_times(self, path, ramp, endpoint):

        if ramp == "linear":
            tf = 1.0 / self.v
            assert path == "both"
            self.ti = tf
            self.gi = 2.0
            self.Ji = 0.0
        elif ramp == "smooth":
            tf = 1.5 * (1.0 / self.v)
            self.ti = tf
            self.gi = 2.0
            assert path == "both"
            self.Ji = 0.0
        
        self._ts = np.array(list(np.arange(-self.ti + self.dt, 0, self.dt)) + [0])
        if not endpoint == "critical":
            self._ts = np.array(list(self._ts) + list(np.arange(self.dt, tf, self.dt)) + [tf])
        
    def _set_params(self):
        
        self.chi = self.options.get('chi', 10)
        self.v = self.options.get('v', 0.1)
        self.gi = self.options.get('gi', 2.0)
        self.gc = self.options.get('gc', 1.0)
        self.dt = self.options.get('dt', 0.1)
        self.path = self.options.get('path', 'both')
        self.ramp = self.options.get('ramp', 'linear')
        self.endpoint = self.options.get('endpoint', 'critical')
        self.evolver = self.options.get('evolver', 'TDVP')
        
        # defaults that need to be set

        self.psi = None
        self.Si = None
        self.E0 = None
        self.g = None
        self.J = None
        self.t = None
        self.Sx = None
        self.Sy = None
        self.Sz = None
#         self.psi0 = None
        
        self.Es = []
        self.Ss = []
        self.Sxs = []
        self.Sys = []
        self.Szs = []
        self.Es_exact = []
        self.gs = []
        self.Js = []
        self.ts = []
        self.chis = []

