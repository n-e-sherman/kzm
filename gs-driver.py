from kzm import KZM
from gs import GroundState
import sys
import os
import subprocess
import multiprocessing
import numpy as np
import pandas as pd
import time


# Main work
def run(options):
    print(options['g'], options['chi'], options['solver'])
    gs = GroundState(options)
    gs.measure()

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
    res_folders = ['gs', 'kzm', 'full']
    for folder in folders:
        if not os.path.isdir(os.getcwd() + '/' + folder):
            os.makedirs(os.getcwd() + '/' + folder)
        if folder == '.results':
        	for res_folder in res_folders:
        		if not os.path.isdir(os.getcwd() + '/' + folder + '/' + res_folder):
		            os.makedirs(os.getcwd() + '/' + folder + '/' + res_folder)

    options = {
        'solver': 'TEBD', # TEBD, DMRG1-bias, DMRG1, DMRG2-bias, DMRG2
        'repo': 'pickle',
        'load': True,
        'J': 1.0,
        'conserve': None
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

    options['dmrg_options'] = dmrg_options
    options['tebd_options'] = tebd_options
    options['repo_options'] = repo_options

    gs = np.linspace(0.9,1.1,201)
    # gs = np.linspace(0.99)
    # gss = [np.linspace(0.8,0.9,1001),np.linspace(0.97,0.98,101),np.linspace(0.96,0.97,101)]
    chis = [2*(1+x) for x in range(15)]
    # chis = [2]
    solvers = ['DMRG2']
    # solvers = ['DMRG1-bias']
    # gs = [1.0]
    # chis = [2,4,6]
    args = []
    for chi in chis:
    # for gs,chi in zip(gss,chis):
        for g in gs:
        	for solver in solvers:
	            arg = options.copy()
	            arg['chi'] = chi
	            arg['g'] = g
	            arg['solver'] = solver
	            args.append(arg)
    p = multiprocessing.Pool(cores)
    p.map(run, args[node::nodes], chunksize=1)
    # [run(arg) for arg in args[node::nodes]]


if __name__ == '__main__':
    main()


