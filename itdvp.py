import numpy as np
import scipy as sp
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.mps import TransferMatrix
from tenpy.networks.mpo import MPO
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import tebd
from scipy.special import ellipe
from tenpy.algorithms import dmrg
import tenpy.linalg.np_conserved as npc

import copy
import logging
import pandas as pd


# probably need to update these import statements.
from tenpy.algorithms.algorithm import TimeEvolutionAlgorithm
# from tenpy.networks.mpo import MPOEnvironment
import tenpy.linalg.np_conserved as npc
from tenpy.tools.params import asConfig
from tenpy.linalg.lanczos import LanczosEvolution
# from tenpy.algorithms.truncation import svd_theta

class iTDVPEngine(TimeEvolutionAlgorithm):
    
    def __init__(self, psi, model, options, **kwargs):
        TimeEvolutionAlgorithm.__init__(self, psi, model, options, **kwargs)
        """
        iTDVP algorithm. We assume that the MPS has a unit cell of 3 sites (do we need to?). 
        Code is based off :cite:`vanderstraeten2019`.
        """
        
        options = self.options
        if model.H_MPO.explicit_plus_hc:
            raise NotImplementedError("TDVP does not respect 'MPO.explicit_plus_hc' flag")
        self.lanczos_options = options.subconfig('lanczos_options')
        self.psi = psi
        self.H = model.calc_H_MPO()
        
#         if environment is None:
#             self.build_environment(model)
            
        self.evolved_time = options.get('start_time', 0.)
        if psi.finite:
            raise ValueError("iTDVP is only implemented for infinite boundary conditions")
        self.L = self.psi.L
        self.dt = options.get('dt', 0.1)
        self.dt_scale = -1.0
        real_time = options.get('real_time',True)
        if real_time:
            self.dt_scale *= 1j
        self.N_steps = options.get('N_steps', 1)
        
    def build_environment(self,model):
        """Builds environment. This currently uses TenPy's stock MPOEnvironment,
        which just performs several iterations of contracting the MPO transfer matrix to
        find the fixed point. Since the system is infinite system size, one could
        find the fixed point exactly by inverting the transfer matrix. Possibly implement 
        this in the future.
        """
        env = MPOEnvironment(self.psi, model.H_MPO, self.psi, start_env_sites = self.options.get('start_env_sites',None))
        self.environment = env
#         self.clean_environment()
        
    @property
    def TDVP_params(self):
        warnings.warn("renamed self.TDVP_params -> self.options", FutureWarning, stacklevel=2)
        return self.options
    
    def run(self):
        """(Real-)time evolution with TDVP.
        """
        active_sites = self.options.get('active_sites', 1)
        if active_sites == 1:
            self.run_one_site(self.N_steps)
        else:
            raise ValueError("iTDVP can only use 1 active sites, not {}".format(active_sites))
    
    def run_one_site(self, N_steps=None):
        """Run the iTDVP algorithm with the one site algorithm.
        .. warning ::
            Be aware that the bond dimension will not increase!
        Parameters
        ----------
        N_steps : integer. Number of steps
        """
        if N_steps != None:
            self.N_steps = N_steps
        for i in range(self.N_steps):
            self.evolved_time = self.evolved_time + self.dt
            self.sweep_one()
            
    
    def sweep_one(self):
        
        LP, RP = make_environment(self.psi, self.H, N=self.options.get('N_env', 20))
        _theta = self.psi.get_B(0,form='Th')
        W = self.H.get_W(0)
        _C = npc.Array.from_ndarray(np.diagflat(self.psi.get_SR(0)), [LP.get_leg('vR').conj(),RP.get_leg('vL').conj()], labels=['vL','vR'])

        dt = self.dt*self.dt_scale
        theta, N_t = self.update_theta_h1(LP,RP,_theta,W,dt) #time evolve theta with H1
        C, N_C = self.update_s_h0(LP,RP,_C,dt) # time evolve s with H0
        
        Cdag = C.conj().itranspose(['vR*','vL*']).iset_leg_labels(['vL','vR'])
        theta_Cdag = npc.tensordot(theta,Cdag,axes=('vR','vL'))
        Cdag_theta = npc.tensordot(Cdag,theta,axes=('vR','vL'))
        A = self.calc_A(theta_Cdag)
        B = self.calc_B(Cdag_theta)
        A, B, theta, s = self.gauge_transform(A,B,theta,C)
        
        self.psi.set_B(0, A, form='A')
        self.psi.set_B(0, theta, form='Th')
        self.psi.set_B(0, B, form='B')
        self.psi.set_SL(0, s)

    def gauge_transform(self, A, B, theta, C):
        
        U, s, Vdag = npc.svd(C, full_matrices=0)
        U = self.set_anonymous_svd(U, 'vR')
        Vdag = self.set_anonymous_svd(Vdag, 'vL')
        Udag = U.conj().itranspose(['vR*','vL*']).iset_leg_labels(['vL','vR'])
        V = Vdag.conj().itranspose(['vR*','vL*']).iset_leg_labels(['vL','vR'])

        Ap = npc.tensordot(Udag,A,axes=('vR','vL'))
        Ap = npc.tensordot(Ap,U,axes=('vR','vL'))
        Bp = npc.tensordot(Vdag,B,axes=('vR','vL'))
        Bp = npc.tensordot(Bp,V,axes=('vR','vL'))
        thetap = npc.tensordot(Udag,theta,axes=('vR','vL'))
        thetap = npc.tensordot(thetap,V,axes=('vR','vL'))
        return Ap, Bp, thetap, s
        
        
            
    def update_theta_h1(self, Lp, Rp, theta, W, dt):
        """Update with the one site Hamiltonian.
        Parameters
        ----------
        Lp : :class:`~tenpy.linalg.np_conserved.Array`
            tensor representing the left environment
        Rp :  :class:`~tenpy.linalg.np_conserved.Array`
            tensor representing the right environment
        theta :  :class:`~tenpy.linalg.np_conserved.Array`
            the theta tensor which needs to be updated
        W : :class:`~tenpy.linalg.np_conserved.Array`
            MPO which is applied to the 'p' leg of theta
        dt : complex number
            time step of the evolution
        """
        H = H1_mixed(Lp, Rp, W)
        theta = theta.combine_legs(['vL', 'p', 'vR'])
        #Initialize Lanczos
        lanczos_h1 = LanczosEvolution(H=H, psi0=theta, options=self.lanczos_options)
        theta_new, N_h1 = lanczos_h1.run(dt)
        res = theta_new.split_legs(['(vL.p.vR)'])
        return res, N_h1
    
    def update_s_h0(self, Lp, Rp, s, dt):
        """Update with the zero site Hamiltonian (update of the singular value)
        Parameters
        ----------
        Lp : :class:`~tenpy.linalg.np_conserved.Array`
            tensor representing the left environment
        Rp :  :class:`~tenpy.linalg.np_conserved.Array`
            tensor representing the right environment
        s : :class:`tenpy.linalg.np_conserved.Array`
            representing the singular value matrix which is updated
        dt : complex number
            time step of the evolution
        """

        H = H0_mixed(Lp,Rp)
        s = s.combine_legs(['vL', 'vR'])
        #Initialize Lanczos
        lanczos_h0 = LanczosEvolution(H=H, psi0=s, options=self.lanczos_options)
        s_new, N_h0 = lanczos_h0.run(dt)
        res = s_new.split_legs(['(vL.vR)'])
        return res, N_h0
    
    def calc_A(self,M):
        """Performs the SVD to extract A from theta * s
        Parameters
        ----------
        M: :class:`tenpy.linalg.np_conserved.Array`
            the tensor to apply svd to
        """
        M = M.combine_legs(['vL', 'p'])
        U, s, V = npc.svd(M, full_matrices=0)
        U = U.split_legs(['(vL.p)'])
        U = self.set_anonymous_svd(U, 'vR')  #U['vL','p','vR']
        V = self.set_anonymous_svd(V, 'vL')  #V['vL','vR']
        new_A = npc.tensordot(U,V,axes=('vR','vL')) #A['vL','p','vR']
        return new_A
    
    def calc_B(self,M):
        """Performs the SVD to extract B from s * theta
        Parameters
        ----------
        M: :class:`tenpy.linalg.np_conserved.Array`
            the tensor to apply svd to
        """
        M = M.combine_legs(['p', 'vR'])
        U, s, V = npc.svd(M, full_matrices=0)
        V = V.split_legs(['(p.vR)'])
        V = self.set_anonymous_svd(V, 'vL') #V['vL','p','vR']
        U = self.set_anonymous_svd(U, 'vR') #U['vL','vR']
        new_B = npc.tensordot(U,V,axes=('vR','vL'))
        return new_B

    def update_AB(self, A, B, theta, C):
        U, s, V = npc.svd(C, full_matrices=0)
        U = self.set_anonymous_svd(U, 'vR')
        V = self.set_anonymous_svd(V, 'vL')
        
        Udag = U.conj()
        Vdag = V.conj()
        
        new_A = npc.tensordot(Udag, A, axes =('vL*', 'vL'))
        new_A = npc.tensordot(new_A, U, axes =('vR', 'vL'))
        new_B = npc.tensordot(V, B, axes = ('vR','vL'))
        new_B = npc.tensordot(new_B, Vdag, axes =('vR', 'vR*'))
        new_theta = npc.tensordot(Udag, theta, axes=('vL*','vL'))
        new_theta = npc.tensordot(new_theta, Vdag, axes=('vR','vR*'))
        new_A.iset_leg_labels(['vL','p','vR'])
        new_B.iset_leg_labels(['vL','p','vR'])
        new_theta.iset_leg_labels(['vL','p','vR'])
        return new_A, new_B, new_theta, s
        
    def set_anonymous_svd(self, U, new_label):
        """Relabel the svd.
        Parameters
        ----------
        U : :class:`tenpy.linalg.np_conserved.Array`
            the tensor which lacks a leg_label
        """
        list_labels = list(U.get_leg_labels())
        for i in range(len(list_labels)):
            if list_labels[i] == None:
                list_labels[i] = 'None'
        U = U.iset_leg_labels(list_labels)
        U = U.replace_label('None', new_label)
        return U
    
class H0_mixed:
    """Class defining the zero site Hamiltonian for Lanczos.
    Parameters
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    Attributes
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    """
    def __init__(self, Lp, Rp):
        self.Lp = Lp
        self.Rp = Rp

    def matvec(self, x):
        x = x.split_legs(['(vL.vR)'])
        x = npc.tensordot(self.Lp, x, axes=('vR', 'vL'))
        x = npc.tensordot(x, self.Rp, axes=(['vR', 'wR'], ['vL', 'wL']))
        #TODO:next line not needed. Since the transpose does not do anything, should not cost anything. Keep for safety ?
        x = x.transpose(['vR*', 'vL*'])
        x = x.iset_leg_labels(['vL', 'vR'])
        x = x.combine_legs(['vL', 'vR'])
        return x


class H1_mixed:
    """Class defining the one site Hamiltonian for Lanczos.
    Parameters
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    M : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p' leg of theta
    Attributes
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    W : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p0' leg of theta
    """
    def __init__(self, Lp, Rp, W):
        self.Lp = Lp  # a,ap,m
        self.Rp = Rp  # b,bp,n
        self.W = W  # m,n,i,ip

    def matvec(self, theta):
        theta = theta.split_legs(['(vL.p.vR)'])
        Lp = self.Lp
        Rp = self.Rp
        W = self.W
        x = npc.tensordot(Lp, theta, axes=('vR', 'vL'))
        x = npc.tensordot(x, W, axes=(['p', 'wR'], ['p*', 'wL']))
        x = npc.tensordot(x, Rp, axes=(['vR', 'wR'], ['vL', 'wL']))
        #TODO:next line not needed. Since the transpose does not do anything, should not cost anything. Keep for safety ?
        x = x.transpose(['vR*', 'p', 'vL*'])
        x = x.iset_leg_labels(['vL', 'p', 'vR'])
        h = x.combine_legs(['vL', 'p', 'vR'])
        return h
    
def make_environment(psi, H, N=5):
    
    IdL = H.get_IdL(0)
    assert IdL is not None
    IdR = H.get_IdR(-1)
    assert IdR is not None
    vL = psi.get_B(0, None).get_leg('vL')
    vR = psi.get_B(0, None).get_leg('vR')
    wL = H.get_W(0).get_leg('wL')
    wR = wL.conj()
    dtype = psi.dtype
    
    LP = npc.diag(1., vR, dtype=dtype, labels=['vR*','vR'])
    LP = LP.add_leg(wR, IdL, axis=1, label='wR')
    RP = npc.diag(1., vL, dtype=dtype, labels=['vL*','vL'])
    RP = RP.add_leg(wL, IdR, axis=1, label='wL')
    A = psi.get_B(0, form='A')
    B = psi.get_B(0, form='B')
    W = H.get_W(0)
    
    TA = npc.tensordot(A, W, axes=('p','p'))
    TA = npc.tensordot(A.conj(), TA, axes=('p*','p*')).itranspose(['vL*','vR*','wL','wR','vL','vR'])
    TB = npc.tensordot(B, W, axes=('p','p'))
    TB = npc.tensordot(B.conj(), TB, axes=('p*','p*')).itranspose(['vL*','vR*','wL','wR','vL','vR'])

    for i in range(N):
        LP = npc.tensordot(LP, TA, axes=(['vR','wR','vR*'],['vL','wL','vL*']))
        RP = npc.tensordot(RP, TB, axes=(['vL','wL','vL*'],['vR','wR','vR*']))
        
    return LP, RP