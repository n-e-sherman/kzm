import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.networks.mps import TransferMatrix
from scipy.special import ellipe

def make_eps(v, ramp='linear'):
    if ramp == 'smooth':
        def eps(t):
            return v*t - (4.0/27.0)*((v*t)**3)
    else:
        def eps(t):
            return v*t
    return eps

def make_gt(gc,eps):
    def gt(t):
        return gc - eps(t)*(2.0 - gc)
    return gt

def make_Jt(eps,path):
    if path == "both":
        def Jt(t):
            return 1 + eps(t)
    else:
        def Jt(t):
            return 0*t + 1
    return Jt

def exact_energy(J, g):
    if not J==0:
        return -J*(2.0/np.pi)*abs(1.0+(g/J))*ellipe(4.0*(g/J)/((1.0+(g/J))**2))
    else:
        return -g

def measure_state(psi, M):
    E = np.mean(M.bond_energies(psi))
    Sx = np.mean(psi.expectation_value('Sx'))
    Sy = np.mean(psi.expectation_value('Sy'))
    Sz = np.mean(psi.expectation_value('Sz'))
    S = np.mean((psi.entanglement_entropy()))
    xi = psi.correlation_length()
    TM = TransferMatrix(psi, psi)
    T, _ = TM.eigenvectors(num_ev=2, which='LM')
    res_dict = {'E': [E], 'S': [S], 
                'Sx': [Sx], 'Sy': [Sy], 'Sz': [Sz], 
                'xi': [xi], 'T1': [abs(T[0])], 'T2': [abs(T[1])]}
    return res_dict

def measure_states(kzm, gs):
    gs_dict = measure_state(gs.psi0, gs.M)
    gs_dict.update(gs.measurement_tags())
    kzm_dict = measure_state(kzm.psi, kzm.M)
    kzm_dict.update(kzm.measurement_tags())
    res_dict = kzm_dict.copy()
    for key in gs_dict:
        res_dict[key+'0'] = gs_dict[key]
    res_dict['l'] = [abs(gs.psi0.overlap(kzm.psi))]
    file = kzm.hash_kzm() + gs.hash_gs()
    return file, res_dict

class uMPS(MPS):

    def __init__(self, sites, Bs, SVs, bc='finite', form='B', norm=1.):
        super().__init__(sites, Bs, SVs, bc=bc, form=form, norm=norm)
        self.B_dict = {}
        self.S = []
        
    def get_B(self, i, form='B', copy=False, cutoff=1E-16, label_p=None):
        if not isinstance(form, str):
            if form in list(self._valid_forms.values()):
                form = list(self._valid_forms.keys())[list(self._valid_forms.values()).index(form)]
        if not form in self.B_dict:
            B = super().get_B(i, form=form, copy=copy, cutoff=cutoff)
        else:
            B = self.B_dict[form]
        if label_p is not None:
            B = self._replace_p_label(B, label_p)
        return B
    
    def set_B(self, i, B, form='B'):
        for j in range(self.L):
            super().set_B(j, B)
        self.B_dict[form] = B
        
    def set_SL(self, i, s):
        for j in range(self.L):
            super().set_SL(j, s)
        self.S = s
        
    def set_SR(self, i, s):
        for j in range(self.L):
            super().set_SR(j, s)
        self.S = s
    
    def get_S(self):
        return self.get_SL(0)
    
    def get_SL(self, i):
        if len(self.S) == 0:
            print('S not stored, creating.')
            S = super().get_SL(i)
            self.S = S
        return self.S
    
    def get_SR(self, i):
        if len(self.S) == 0:
            print('S not stored, creating.')
            S = super().get_SR(i)
            self.S = S
        return self.S
        
    def set_B_dict(self, Bs):
        for k,v in Bs.items():
            for j in range(self.L):
                super().set_B(j, v, form=k)
            self.B_dict[k] = v
            
    def copy(self):
        res = super().copy()
        res.B_dict = self.B_dict
        return res
    
    
def make_uMPS(psi_in):
    return uMPS(psi_in.sites, psi_in._B, psi_in._S, bc='infinite', form='B', norm=1.0)

def make_uniform(psi):
    _chis = psi.chi

    #RCF
    TB = TransferMatrix(psi, psi, shift_bra=1, shift_ket=0)
    es_B, evs_B = TB.eigenvectors()
    U = evs_B[0].split_legs(['(vL.vL*)']).iset_leg_labels(['vL','vR'])
    _B = psi.get_B(1)
    B = npc.tensordot(_B, U, axes=('vR','vL'))
    B = (B/npc.norm(B))*np.sqrt(_chis[1])

    #LCF
    TA = TransferMatrix(psi, psi, shift_bra=1, shift_ket=0, form='A', transpose=True)
    es_A, evs_A = TA.eigenvectors()
    V = evs_A[0].split_legs(['(vR*.vR)'])
    Vdag = V.conj()
    _A = psi.get_B(1, form='A')
    A = npc.tensordot(_A, Vdag, axes=('vR','vR*'))
    A = (A/npc.norm(A))*np.sqrt(_chis[1])

    s = 0.5*(psi.get_SL(0) + psi.get_SL(1))
    upsi = make_uMPS(psi)
    upsi.set_B(0, A, form='A')
    upsi.set_B(0, B)
    upsi.set_SL(0, s)
    return upsi


