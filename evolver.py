from tenpy.algorithms import tebd

from itdvp import *
from util import *

class Evolver:
    
    def __init__(self, options):
        self.options = options

class TEBD_Evolver(Evolver):
    
    first = True
    def __init__(self, options):
        super().__init__(options)
        
    def evolve(self, psi, M):
        eng = tebd.TEBDEngine(psi, M, self.options)
        eng.run()
        return psi

class TDVP_Evolver(Evolver):
    
    first = True
    def __init__(self, options, options_tebd, chi):
        super().__init__(options)
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