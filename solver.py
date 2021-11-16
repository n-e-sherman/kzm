import numpy as np
from tenpy.algorithms import tebd
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import dmrg

from tenpy.models.lattice import Site, Chain
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
class TFIChainField(TFIChain):
  
    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.))
        g = np.asarray(model_params.get('g', 1.))
        h = np.asarray(model_params.get('h', 0.1))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmaz')
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-h, u, 'Sigmax')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmax', u2, 'Sigmax', dx)

class TFIChainParity(TFIChain):
  
    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.))
        g = np.asarray(model_params.get('g', 1.))
        h = np.asarray(model_params.get('h', 0.1))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmaz')
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-h, u, 'Sigmax')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmax', u2, 'Sigmax', dx)
class Solver:
    
    def __init__(self, options):
        self.options = options
        
class DMRG1_Solver(Solver):
    
    def __init__(self, options):
        super().__init__(options)
    
    def solve(self, model_params, psi):
        M = TFIChain(model_params)
        eng = dmrg.SingleSiteDMRGEngine(psi, M, self.options)
        return eng.run()
    
class DMRG1_bias_Solver(Solver):
    
    def __init__(self, options):
        super().__init__(options)
    
    def solve(self, model_params, psi):
        field_params1 = model_params.copy()
        field_params2 = model_params.copy()
        field_params1['h'] = 0.1
        field_params2['h'] = 0.01
        M = TFIChain(model_params)
        Mb1 = TFIChainField(field_params1)
        Mb2 = TFIChainField(field_params2)
        eng = dmrg.SingleSiteDMRGEngine(psi, Mb1, self.options)
        _, psi = eng.run()
        eng = dmrg.SingleSiteDMRGEngine(psi, Mb2, self.options)
        _, psi = eng.run()
        eng = dmrg.SingleSiteDMRGEngine(psi, M, self.options)
        return eng.run()
        
class DMRG2_Solver(Solver):
    
    def __init__(self, options):
        super().__init__(options)
        
    def solve(self, model_params, psi):
        M = TFIChain(model_params)
        eng = dmrg.TwoSiteDMRGEngine(psi, M, self.options)
        return eng.run()
        
class DMRG2_bias_Solver(Solver):
    
    def __init__(self, options):
        super().__init__(options)
        
    def solve(self, model_params, psi):
        field_params1 = model_params.copy()
        field_params2 = model_params.copy()
        field_params1['h'] = 0.1
        field_params2['h'] = 0.01
        M = TFIChain(model_params)
        Mb1 = TFIChainField(field_params1)
        Mb2 = TFIChainField(field_params2)
        eng = dmrg.TwoSiteDMRGEngine(psi, Mb1, self.options)
        _, psi = eng.run()
        eng = dmrg.TwoSiteDMRGEngine(psi, Mb2, self.options)
        _, psi = eng.run()
        eng = dmrg.TwoSiteDMRGEngine(psi, M, self.options)
        return eng.run()
        
class TEBD_Solver(Solver):
    
    def __init__(self, options):
        super().__init__(options)
        
    def solve(self, model_params, psi):
        M = TFIChain(model_params)
        eng = tebd.TEBDEngine(psi, M, self.options)
        eng.run_GS()
        return np.mean(M.bond_energies(psi)), psi