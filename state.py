from repository import *
from typing import Any

class State:
    
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
        self.dt = options.get('dt', 0.1)
        self.chi = options.get('chi', 10)
        self._set_options()
        self._set_repo()
        
    def _hash(self, label, hash_dict):
        res = label
        for k,v in hash_dict.items():
            res += '-'+k+'='+str(v)
        return res
    
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
        
        self.repo_options = {
            
        }
        
        self.dmrg_options = {'mixer': True, 
                             'trunc_params': { 'chi_max': self.chi, 'svd_min': 1.e-10 } , 
        }
        
        self.tebd_options.update(self.options.get('tebd_options',{}))
        self.tdvp_options.update(self.options.get('tdvp_options',{}))
        self.repo_options.update(self.options.get('repo_options',{}))
        self.dmrg_options.update(self.options.get('dmrg_options',{}))
        
        
    def _set_repo(self):
        self.repo = Repository_Pickle(self.repo_options)

