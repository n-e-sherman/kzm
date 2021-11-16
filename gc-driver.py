from kzm import KZM
from gs import GroundState
import sys
import os
import subprocess
import multiprocessing
import numpy as np
import scipy as sp
import pandas as pd
import time

def make_f_Q(options, Q):
    def f_Q(g):
        options['g'] = float(g)
        gs = GroundState(options)
        gs.measure()
        return -1.0*abs(float(gs.measurements[Q].unique()))
    return f_Q

# Main work
def run(options):
    
    print(options['chi'], options['solver'], options['Q'])
    g0 = options['g0']
    gmin = 0.97*g0
    gmax = 1.03*g0
    Q = options['Q']
    
    f = make_f_Q(options, Q)
    res = sp.optimize.minimize(f, [g0], bounds = [(gmin,gmax)])
    Q_value = f(res.x)
    df = pd.DataFrame({'gc': res.x, 'Q': [Q], 'solver': [options['solver']], 'conserve': [options['conserve']], 'chi': [options['chi']]})
    df.to_csv('.results/gc/gc:-Q='+Q+'-chi='+str(options['chi'])+'-solver='+options['solver']+'-conserve='+str(options['conserve'])+'.csv', index=False)
    print(res)
    print('measurement of ' + Q + ':', abs(Q_value),'gc:', res.x)


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
    res_folders = ['gs', 'kzm', 'full','gc']
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
        'conserve': 'parity',
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
    
    dfgc = pd.read_csv('gc.csv')

    options['dmrg_options'] = dmrg_options
    options['tebd_options'] = tebd_options
    options['repo_options'] = repo_options


    chis = [2*(1+x) for x in range(15)]
    
    # chis = [4]
    Qs = ['S','xi']
    solvers = ['DMRG2']
    conserves = ['None']
    
    args = []
    for chi in chis:
        for Q in Qs:
            for solver,conserve in zip(solvers,conserves):
                arg = options.copy()
                g0 = float(dfgc.loc[(dfgc['Q'] == Q) & (dfgc['conserve'] == conserve) & (dfgc['chi'] == chi)].gc.unique())
                arg['g0'] = g0
                arg['chi'] = chi
                arg['solver'] = solver
                arg['conserve'] = conserve
                arg['Q'] = Q
                if conserve == "None":
                    arg['conserve'] = None
                args.append(arg)
#     p = multiprocessing.Pool(cores)
#     p.map(run, args[node::nodes], chunksize=1)
    [run(arg) for arg in args[node::nodes]]


# if __name__ == '__main__':
main()




