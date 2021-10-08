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
        'chi': 4,
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

    dmrg_options = {
        
    }

    # '/global/cscratch1/sd/nsherman/'
    repo_options = {
        'data_dir' : os.getcwd() + '/.data/',
        'results_dir' : os.getcwd() + '/.results/',
        'log_dir' : os.getcwd() + '/.log/'
    }

    options['tdvp_options'] = tdvp_options
    options['tebd_options'] = tebd_options
    options['dmrg_options'] = dmrg_options
    options['repo_options'] = repo_options

    dfgc = pd.read_csv('gc.csv')
    vi = 1E-5
    vf = 0.5
    Nv = 50
    x = np.log(vf/vi)/(Nv-1)
    vs = [vi*np.exp(n*x) for n in range(Nv)][::-1]
    chis = [2*(1+x) for x in range(10)]

    args = multiprocessing.Queue()
    args = []
    for v in vs:
        for chi in chis:
            arg = options.copy()
            gc = float(dfgc.loc[dfgc['chi'] == chi].g)
            arg['gc'] = gc
            arg['chi'] = chi
            arg['v'] = v
            args.append(arg)
    p = multiprocessing.Pool(cores)
    p.map(run, args[node::nodes], chunksize=1)

if __name__ == '__main__':
    main()
