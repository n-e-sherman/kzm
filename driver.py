from kzm import KZM
import sys
import os
import subprocess
import multiprocessing
import numpy as np



###########################
########## setup ##########
###########################

cores = 1
nodes = 1
node = 0
if len(sys.argv) > 1:
    cores = int(sys.argv[1])
if len(sys.argv) > 2:
    nodes = int(sys.argv[2])
if len(sys.argv) > 3:
    node = int(sys.argv[3])
print(cores, nodes, node)

folders = ['.data', '.log', '.results']
for folder in folders:
    if not os.path.isdir(os.getcwd() + '/' + folder):
        os.makedirs(os.getcwd() + '/' + folder)




# Main work
def run(options):
    kzm = KZM(options)
    kzm.evolve()

if __name__ == '__main__':

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

    vi = 1E-5
    vf = 0.5
    Nv = 50
    x = np.log(vf/vi)/(Nv-1)
    vs = [vi*np.exp(n*x) for n in range(Nv)][::-1]
    chis = [2*(1+x) for x in range(10)]

    args = []
    for v in vs:
        for chi in chis:
            arg = options.copy()
            arg['chi'] = chi
            arg['v'] = v
            args.append(arg)
    p = multiprocessing.Pool(cores)
    p.map(run, args[node::nodes])