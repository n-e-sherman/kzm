from kzm import KZM
import os
import numpy as np



vi = 1E-5
vf = 0.5
Nv = 50
x = np.log(vf/vi)/(Nv-1)
vs = [vi*np.exp(n*x) for n in range(Nv)][::-1]
chis = [2*(1+x) for x in range(10)]

# testing
# vs = [1]
# chis = [2]


options = {
    'ramp' : 'linear', # [linear, smooth]
    'path' : 'g', # [both, g]
    'endpoint' : 'critical', # [critical, ising]
    'gi' : 3,
    'v' : 0.1,
    'dt' : 0.1,
    'chi': 4,
    'evolver': 'TDVP',
    'repo': 'pickle',
    'load': True
}

tdvp_options = {
    
}

tebd_options = {
    'order': 2,
}

dmrg_options = {
    
}

repo_options = {
    'cwd' : os.getcwd(),
    'data_dir' : '.data/',
    'res_dir' : '.results/',
    'log_dir' : '.log/'
}

options['tdvp_options'] = tdvp_options
options['tebd_options'] = tebd_options
options['dmrg_options'] = dmrg_options
options['repo_options'] = repo_options

for v in vs:
    for chi in chis:
        options['chi'] = chi
        options['v'] = v
        kzm = KZM(options)
        kzm.evolve()