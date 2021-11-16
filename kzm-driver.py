from kzm import KZM
import sys
import os
import subprocess
import multiprocessing
import numpy as np
import pandas as pd
import time


# Main work
def run(options):
    print(options['v'], options['chi'])
    kzm = KZM(options)
    kzm.evolve()
    kzm.measure()

def main():

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
            
    options = {
        'ramp' : 'linear', # [linear, smooth]
        'path' : 'both', # [both, g]
        'endpoint' : 'critical', # [critical, ising]
        'gi' : 3,
        'v' : 0.1,
        'dt' : 0.1,
        'gc': 1.0, # This needs to be chi dependent, could hard code it in?
        'evolver': 'TEBD',
        'repo': 'pickle',
        'load': True
    }
    
    tdvp_options = {
        
    }

    tebd_options = {
        'order': 2,
    }

    # '/global/cscratch1/sd/nsherman/'
    repo_options = {
        'data_dir' : os.getcwd() + '/.data/',
        'results_dir' : os.getcwd() + '/.results/',
        'log_dir' : os.getcwd() + '/.log/'
    }

    options['tdvp_options'] = tdvp_options
    options['tebd_options'] = tebd_options
    options['repo_options'] = repo_options

    vi = 1E-5
    vf = 0.5
    Nv = 50
    x = np.log(vf/vi)/(Nv-1)
    vs = [vi*np.exp(n*x) for n in range(Nv)][::-1]
    chis = [2*(1+x) for x in range(10)]
    # vs = [0.5]
    # chis = [2,4,6]

    dfgc = pd.read_csv('gc.csv')
    args = []
    ramps = ['smooth']
    for v in vs:
        for chi in chis:
            for ramp in ramps:
                gc = float(dfgc.loc[(dfgc['chi'] == chi) & (dfgc['conserve'] == 'parity') & (dfgc['Q'] == 'S') & (dfgc['solver'] == 'DMRG2')].gc.unique())
                arg = options.copy()
                arg['gc'] = gc
                arg['chi'] = chi
                arg['v'] = v
                arg['ramp'] = ramp
                args.append(arg)
    p = multiprocessing.Pool(cores)
    p.map(run, args[node::nodes], chunksize=1)
    # [run(arg) for arg in args[node::nodes]]

if __name__ == '__main__':
    main()


