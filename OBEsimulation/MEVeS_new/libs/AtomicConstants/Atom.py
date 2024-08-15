import numpy as np
import os
import pandas as pd
from scipy.constants import physical_constants, epsilon_0, hbar, c, Boltzmann
from scipy.linalg import kron, eigh
from sympy.physics.quantum.cg import Wigner6j
from sympy.physics.quantum.cg import CG
from fractions import Fraction
amu = physical_constants['atomic mass constant'][0] #An atomic mass unit in kg
au = physical_constants['atomic unit of electric dipole mom.'][0] # atomic units for reduced dipole moment

SUB_STRING = 'MEVeS_new'

index = os.path.abspath('').find(SUB_STRING)

if index == -1:
    DIRECTORY = os.path.abspath('')
else:
    DIRECTORY = os.path.abspath('')[:os.path.abspath('').find(SUB_STRING) + len(SUB_STRING)]

class Alkalis:
    def __init__(self):
        self.S = 1/2
        self.Lg = 0
        self.Jg = 1/2
        self.Lj = 1

        self.unpack_config()

        self.lookup_atom_constants()

        # hyperfine splitting - calculate?
        self.hyperfine_splitting()

        # coupling tensor - separate ones for ge, es, etc?
        self.coupling_ge = self.coupling(self.Jg, self.Fg, self.mg, self.Jj, self.Fj, self.mj)
        self.coupling_es = self.coupling(self.Jj, self.Fj, self.mj, self.Jq, self.Fq, self.mq)
        # dressing
        if self.number_of_states == 4:
            self.coupling_sb = self.coupling(self.Jq, self.Fq, self.mq, self.Jb, self.Fb, self.mb)
        if self.number_of_states == 5: # ground state mapping or dressing with two states
            # dressing with two states
            if 'dressing2' in self.config["states"].keys():
                self.coupling_sb = self.coupling(self.Jq, self.Fq, self.mq, self.Jb, self.Fb, self.mb) # assuming 'dresssing' exists if 'dressing2' exists
                self.coupling_sb2 = self.coupling(self.Jq, self.Fq, self.mq, self.Jb2, self.Fb2, self.mb2)
            # ground state mapping
            else:
                self.coupling_be = self.coupling(self.Jb, self.Fb, self.mb, self.Jj, self.Fj, self.mj)

        if self.number_of_states == 6: # DRAGON
            self.coupling_be = self.coupling(self.Jb, self.Fb, self.mb, self.Jj, self.Fj, self.mj)
            self.coupling_sd = self.coupling(self.Jd, self.Fd, self.md, self.Jq, self.Fq, self.mq)

    def __repr__(self):
        ground_state = self.ground_state.replace('/', 'q') # use q instead of / for use in file names
        intermediate_state = self.intermediate_state.replace('/', 'q')
        storage_state = self.storage_state.replace('/', 'q')
        if self.config["states"]["storage"]["L"] == 0: # using ground state to store i.e. lambda scheme
            string = f"{self.atom}_{ground_state}F{self.Fg}_{intermediate_state}_{storage_state}F{self.Fq}"
        else:
            string = f"{self.atom}_{ground_state}F{self.Fg}_{intermediate_state}_{storage_state}"
        # dressing state
        if self.number_of_states == 4:
            string += f"_{self.dressing_state.replace('/', 'q')}"

        # ground state mapping
        if self.number_of_states == 5:
            # dressing with two states
            if 'dressing2' in self.config["states"].keys():
                string += f"_{self.dressing_state.replace('/', 'q')}"
                string += f"_{self.dressing_state2.replace('/', 'q')}"
            else:
                string += f"_{intermediate_state}_{ground_state}F{self.Fb}"

        if self.splitting:
            string += f"_HyperfineSplittingYes"
        else:
            string += f"_HyperfineSplittingNo"
        
        return string 

    
    def hyperfine_states(self, I, J):
        Fmin = np.abs(J - I)
        Fmax = np.abs(J + I)
        F = np.arange(Fmin, Fmax+1)
        return F
    
    def momentum_letter(self, L):
        if L == 0:
            return 's'
        elif L == 1:
            return 'p'
        elif L == 2:
            return 'd'
        elif L == 3:
            return 'f'
    
    def unpack_config(self):
        
        self.number_of_states = len(self.config["states"])
        self.splitting = self.config["Hyperfine splitting"]

        self.nj = self.config["states"]["intermediate"]["n"]
        self.Jj = self.config["states"]["intermediate"]["J"]

        self.nq = self.config["states"]["storage"]["n"]
        self.Lq = self.config["states"]["storage"]["L"]
        if self.nq==self.ng and self.Lq==self.Lg: # using ground state to store i.e. lambda scheme
            self.Jq = self.Jg
        else:
            self.Jq = self.config["states"]["storage"]["J"]
            

        # dressing state
        if self.number_of_states == 4:
            self.nb = self.config["states"]["dressing"]["n"]
            self.Lb = self.config["states"]["dressing"]["L"]
            self.Jb = self.config["states"]["dressing"]["J"]

        if self.number_of_states == 5: # ground state mapping or dressing with two states
            if 'dressing2' in self.config["states"].keys(): # dressing with two states
                self.nb = self.config["states"]["dressing"]["n"]
                self.Lb = self.config["states"]["dressing"]["L"]
                self.Jb = self.config["states"]["dressing"]["J"]
                self.nb2 = self.config["states"]["dressing2"]["n"]
                self.Lb2 = self.config["states"]["dressing2"]["L"]
                self.Jb2 = self.config["states"]["dressing2"]["J"]

            else: # ground state mapping
                # assume intermediate excited state the same for mapping up to mapping down
                self.nb = self.ng #self.config["states"]["dressing"]["n"]
                self.Lb = self.Lg #self.config["states"]["dressing"]["L"]
                self.Jb = self.Jg #self.config["states"]["dressing"]["J"]

        if self.number_of_states == 6: # ground state mapping with rephasing (DRAGON)
            self.nb = self.ng 
            self.Lb = self.Lg 
            self.Jb = self.Jg

            self.nd = self.config["states"]["dressing"]["n"]
            self.Ld = self.config["states"]["dressing"]["L"]
            self.Jd = self.config["states"]["dressing"]["J"]




        if self.splitting == False:
            self.Fg = np.array([0])
            self.mg = np.array([0])

            self.Fj = np.array([0])
            self.mj = np.array([0])

            self.Fq = np.array([0])
            self.mq = np.array([0])

            # dressing state
            if self.number_of_states == 4:
                self.Fb = np.array([0])
                self.mb = np.array([0])         

            # ground state mapping 
            if self.number_of_states == 5:
                if 'dressing2' in self.config["states"].keys(): # dressing with two states
                    self.Fb = np.array([0])
                    self.mb = np.array([0])         
                    self.Fb2 = np.array([0])
                    self.mb2 = np.array([0])  
                else: # ground state mapping
                    self.Fb = np.array([0])
                    self.mb = np.array([0]) 

            # DRAGON 
            if self.number_of_states == 6:
                self.Fb = np.array([0])
                self.mb = np.array([0]) 

                self.Fd = np.array([0])
                self.md = np.array([0]) 

        else: # hyperfine splitting
            # Ground state
            self.Fg = np.array([self.config["states"]["initial"]["F"]])
            self.mg = np.arange(-max(self.Fg), max(self.Fg)+1)

            # intermediate state
            self.Fj = self.hyperfine_states(self.I, self.Jj)
            self.mj = np.arange(-max(self.Fj), max(self.Fj)+1)

            # storage state
            if self.nq==self.ng and self.Lq==self.Lg: # using ground state to store i.e. lambda scheme
                # could have same F as initial state, need to be pumped into initial mF state!
                self.Fq = np.array([self.config["states"]["storage"]["F"]])
                self.mq = np.arange(-max(self.Fq), max(self.Fq)+1)
            else:
                self.Fq = self.hyperfine_states(self.I, self.Jq)
                self.mq = np.arange(-max(self.Fq), max(self.Fq)+1)

            # dressing state
            if self.number_of_states == 4:
                self.Fb = self.hyperfine_states(self.I, self.Jb)
                self.mb = np.arange(-max(self.Fb), max(self.Fb)+1)

            # ground state mapping
            if self.number_of_states == 5:
                if 'dressing2' in self.config["states"].keys(): # dressing with two states
                    self.Fb = self.hyperfine_states(self.I, self.Jb)
                    self.mb = np.arange(-max(self.Fb), max(self.Fb)+1)
                    self.Fb2 = self.hyperfine_states(self.I, self.Jb2)
                    self.mb2 = np.arange(-max(self.Fb2), max(self.Fb2)+1)
                else: # ground state mapping
                    self.Fb = np.array([self.config["states"]["storage2"]["F"]])
                    self.mb = np.arange(-max(self.Fb), max(self.Fb)+1)

            # DRAGON
            if self.number_of_states == 6:
                self.Fb = np.array([self.config["states"]["storage2"]["F"]])
                self.mb = np.arange(-max(self.Fb), max(self.Fb)+1)

                self.Fd = np.array([self.config["states"]["dressing"]["F"]])
                self.md = np.arange(-max(self.Fd), max(self.Fd)+1)

    def lookup_atom_constants(self):
        self.ground_state = str(self.ng)+self.momentum_letter(self.Lg)+str(Fraction(self.Jg))
        self.intermediate_state = str(self.nj)+self.momentum_letter(self.Lj)+str(Fraction(self.Jj))
        self.storage_state = str(self.nq)+self.momentum_letter(self.Lq)+str(Fraction(self.Jq))

        df = pd.read_csv(self.filename)

        wavelengths = np.zeros(self.number_of_states-1)
        reduced_dipoles = np.zeros(self.number_of_states-1)
        self.reduced_dipoles = np.zeros(self.number_of_states - 1)
        self.lifetimes = np.zeros(self.number_of_states - 1)

        self.J_array = np.zeros(self.number_of_states)
        self.J_array[0] = self.Jg
        self.J_array[1] = self.Jj
        self.J_array[2] = self.Jq

        # make into loop?
        query = df[
                        (df['Initial'].str.contains(self.ground_state) & df['Final'].str.contains(self.intermediate_state)) |
                        (df['Final'].str.contains(self.ground_state) & df['Initial'].str.contains(self.intermediate_state))
                    ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

        [[wavelengths[0], reduced_dipoles[0]]] = query

        query = df[
                        (df['Initial'].str.contains(self.intermediate_state) & df['Final'].str.contains(self.storage_state)) |
                        (df['Final'].str.contains(self.intermediate_state) & df['Initial'].str.contains(self.storage_state))
                    ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

        [[wavelengths[1], reduced_dipoles[1]]] = query

        self.reduced_dipoles[0] = reduced_dipoles[0]*au/np.sqrt(2*self.J_array[0]+1) # to make same convention as Steck
        self.reduced_dipoles[1] = reduced_dipoles[1]*au/np.sqrt(2*self.J_array[1]+1)

        if self.number_of_states == 4:
            # dressing
            self.dressing_state = str(self.nb)+self.momentum_letter(self.Lb)+str(Fraction(self.Jb))
            query = df[
                        (df['Initial'].str.contains(self.storage_state) & df['Final'].str.contains(self.dressing_state)) |
                        (df['Final'].str.contains(self.storage_state) & df['Initial'].str.contains(self.dressing_state))
                    ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

            [[wavelengths[2], reduced_dipoles[2]]] = query
            self.reduced_dipoles[2] = reduced_dipoles[2]*au/np.sqrt(2*self.J_array[2]+1)
            self.J_array[3] = self.Jb


        elif self.number_of_states == 5:
            if 'dressing2' in self.config["states"].keys(): # dressing with two states
                # dressing
                self.dressing_state = str(self.nb)+self.momentum_letter(self.Lb)+str(Fraction(self.Jb))
                query = df[
                            (df['Initial'].str.contains(self.storage_state) & df['Final'].str.contains(self.dressing_state)) |
                            (df['Final'].str.contains(self.storage_state) & df['Initial'].str.contains(self.dressing_state))
                        ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

                [[wavelengths[2], reduced_dipoles[2]]] = query
                self.reduced_dipoles[2] = reduced_dipoles[2]*au/np.sqrt(2*self.J_array[2]+1)
                self.J_array[3] = self.Jb
                # dressing 2
                self.dressing_state2 = str(self.nb2)+self.momentum_letter(self.Lb2)+str(Fraction(self.Jb2))
                query = df[
                        (df['Initial'].str.contains(self.storage_state) & df['Final'].str.contains(self.dressing_state2)) |
                        (df['Final'].str.contains(self.storage_state) & df['Initial'].str.contains(self.dressing_state2))
                    ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()
                [[wavelengths[3], reduced_dipoles[3]]] = query
                self.reduced_dipoles[3] = reduced_dipoles[3]*au/np.sqrt(2*self.J_array[3]+1)
                self.J_array[4] = self.Jb2
            else:
                # ground state mapping
                # mapping field 1 has the same wavelength and dipole as control field (for ORCA)
                [wavelengths[2], reduced_dipoles[2]] = [wavelengths[1], reduced_dipoles[1]]
                self.reduced_dipoles[2] = reduced_dipoles[2]*au/np.sqrt(2*self.J_array[1]+1) # (j -> q)
                self.J_array[3] = self.Jj
                # mapping field 2 has the same wavelength and dipole as signal field (for ORCA)
                [wavelengths[3], reduced_dipoles[3]] = [wavelengths[0], reduced_dipoles[0]]
                self.reduced_dipoles[3] = reduced_dipoles[3]*au/np.sqrt(2*self.J_array[0]+1) # (g -> j)
                self.J_array[4] = self.Jg

        elif self.number_of_states == 6: # DRAGON
            # ground state mapping
            # mapping field 1 has the same wavelength and dipole as control field (for ORCA)
            [wavelengths[2], reduced_dipoles[2]] = [wavelengths[1], reduced_dipoles[1]]
            self.reduced_dipoles[2] = reduced_dipoles[2]*au/np.sqrt(2*self.J_array[1]+1) # (j -> q)
            self.J_array[3] = self.Jj
            # mapping field 2 has the same wavelength and dipole as signal field (for ORCA)
            [wavelengths[3], reduced_dipoles[3]] = [wavelengths[0], reduced_dipoles[0]]
            self.reduced_dipoles[3] = reduced_dipoles[3]*au/np.sqrt(2*self.J_array[0]+1) # (g -> j)
            self.J_array[4] = self.Jg
            # dressing field
            self.dressing_state = str(self.nd)+self.momentum_letter(self.Ld)+str(Fraction(self.Jd))
            query = df[
                        (df['Initial'].str.contains(self.storage_state) & df['Final'].str.contains(self.dressing_state)) |
                        (df['Final'].str.contains(self.storage_state) & df['Initial'].str.contains(self.dressing_state))
                    ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

            [[wavelengths[4], reduced_dipoles[4]]] = query
            self.reduced_dipoles[4] = reduced_dipoles[4]*au/np.sqrt(2*self.J_array[4]+1)
            self.J_array[5] = self.Jd
            

        self.wavelengths = wavelengths*1e-9 # convert to m
        self.angular_frequencies = 2*np.pi*c/self.wavelengths
        self.wavevectors = 2*np.pi/self.wavelengths

        self.lifetimes = ( 3*np.pi*epsilon_0*hbar*pow(c,3) * 
             (2*self.J_array[1:] + 1)/(pow(self.angular_frequencies,3)*(2*self.J_array[:-1] + 1)*pow(self.reduced_dipoles,2)) )
        
        if self.number_of_states == 5:
            if 'dressing2' not in self.config["states"].keys(): # if not dressing with two states
                # ground state mapping
                # bit of a hack - find a better way to do this
                self.lifetimes[2] = self.lifetimes[0]

        if self.number_of_states == 6:
            # ground state mapping
            # bit of a hack - find a better way to do this
            self.lifetimes[2] = self.lifetimes[0]
        
        self.decay_rates = 1/self.lifetimes
        
        if self.Lq == 0: # storage state is in ground state manifold
            self.decay_rates[1] = 0 # set spin wave decay to be zero
        elif self.number_of_states == 5:
            if 'dressing2' not in self.config["states"].keys(): # if not dressing with two states
                self.decay_rates[3] = 0 # set spin wave decay of ground state to be zero
        elif self.number_of_states == 6:
            self.decay_rates[3] = 0 # set spin wave decay of ground state to be zero

        self.gammas = self.decay_rates/2   

    def Hhfs(self, J, I):
        """Provides the I dot J matrix (hyperfine structure interaction)"""
        gJ=int(2*J+1)
        Jx=self.jx(J)
        Jy=self.jy(J)
        Jz=self.jz(J)
        Ji=np.identity(gJ)
        J2=np.dot(Jx,Jx)+np.dot(Jy,Jy)+np.dot(Jz,Jz)

        gI=int(2*I+1)
        gF=gJ*gI
        Ix=self.jx(I)
        Iy=self.jy(I)
        Iz=self.jz(I)
        Ii=np.identity(gI)
        Fx=kron(Jx,Ii)+kron(Ji,Ix)
        Fy=kron(Jy,Ii)+kron(Ji,Iy)
        Fz=kron(Jz,Ii)+kron(Ji,Iz)
        Fi=np.identity(gF)
        F2=np.dot(Fx,Fx)+np.dot(Fy,Fy)+np.dot(Fz,Fz)
        Hhfs=0.5*(F2-I*(I+1)*Fi-kron(J2,Ii))
        return Hhfs

    def Bbhfs(self, J, I):
        """Calculates electric quadrupole matrix.
        """
        gJ=int(2*J+1)
        Jx=self.jx(J)
        Jy=self.jy(J)
        Jz=self.jz(J)

        gI=int(2*I+1)
        gF=gJ*gI
        Ix=self.jx(I)
        Iy=self.jy(I)
        Iz=self.jz(I)
        
        Fi=np.identity(gF)

        IdotJ=kron(Jx,Ix)+kron(Jy,Iy)+kron(Jz,Iz)
        IdotJ2=np.dot(IdotJ,IdotJ)

        if I != 0:
            Bbhfs=1./(6*I*(2*I-1))*(3*IdotJ2+3./2*IdotJ-I*(I+1)*15./4*Fi)
        else:
            Bbhfs = 0
        return Bbhfs

    def _jp(self, jj):
        b = 0
        dim = int(2*jj+1)
        jp = np.zeros((dim,dim))
        z = np.arange(dim)
        m = jj-z
        while b<dim-1:
            mm = m[b+1]
            jp[b,b+1] = np.sqrt(jj*(jj+1)-mm*(mm+1)) 
            b = b+1
        return jp

    def jx(self, jj):
        jp = self._jp(jj)
        jm=np.transpose(jp)
        jx=0.5*(jp+jm)
        return jx

    def jy(self, jj):
        jp = self._jp(jj)
        jm=np.transpose(jp)
        jy=0.5j*(jm-jp)
        return jy

    def jz(self, jj):
        jp = self._jp(jj)
        jm=np.transpose(jp)
        jz=0.5*(np.dot(jp,jm)-np.dot(jm,jp))
        return jz

    def hyperfine_splitting(self):
        # calculate from hyperfine constant?
        # or just lookup?

        # ground state hyperfine splitting
        df = pd.read_csv(self.filename_hyperfine)

        Ag = df[df['State'].str.contains(self.ground_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
        Bg = df[df['State'].str.contains(self.ground_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6

        H = Ag*self.Hhfs(self.Jg, self.I)
        values = eigh(H)[0].real
        self.deltaHF = abs(values[-1] - values[0])
        if self.config["states"]["initial"]["F"] == self.hyperfine_states(self.I, self.Jg)[1]:
            self.deltaHF *= -1
            
        if self.splitting == False:
            self.wg = np.array([0])
            self.wj = np.array([0])
            self.wq = np.array([0])
            if self.number_of_states == 4:
                self.wb = np.array([0])
            if self.number_of_states == 5:
                if 'dressing2' in self.config["states"].keys(): # dressing with two states
                    self.wb = np.array([0])
                    self.wb2 = np.array([0])
                    self.dress_state_splitting = (self.angular_frequencies[2] - self.angular_frequencies[3])/(2*np.pi)
                else:
                    # ground state mapping
                    self.wb = np.array([0])
            elif self.number_of_states == 6:
                self.wb = np.array([0])
                self.wd = np.array([0])

        else:
            df = pd.read_csv(self.filename_hyperfine)

            Ag = df[df['State'].str.contains(self.ground_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
            Bg = df[df['State'].str.contains(self.ground_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6

            Aj = df[df['State'].str.contains(self.intermediate_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
            Bj = df[df['State'].str.contains(self.intermediate_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6

            Aq = df[df['State'].str.contains(self.storage_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
            Bq = df[df['State'].str.contains(self.storage_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6

            if len(self.Fg)>1:
                H = Ag*self.Hhfs(self.Jg, self.I)
                values = eigh(H)[0].real
                indices = np.concatenate(([0], np.cumsum(2*self.Fg[:-1]+1))).astype(int)
                if Ag < 0:
                    self.wg = np.flip(values)[indices]
                else:
                    self.wg = values[indices]
            else:
                self.wg = np.array([0])


            if len(self.Fj)>1:
                dim = int((2*self.Lj+1)*(2*self.S+1)*(2*self.I+1))  # total dimension of matrix
                H = Aj * self.Hhfs(self.Jj, self.I) + Bj*self.Bbhfs(self.Jj,self.I)
                values = eigh(H)[0].real
                indices = np.concatenate(([0], np.cumsum(2*self.Fj[:-1]+1))).astype(int)
                if Aj < 0:
                    self.wj = np.flip(values)[indices]
                else:
                    self.wj = values[indices]
                
            else:
                self.wj = np.array([0])

            if len(self.Fq)>1:
                H = Aq * self.Hhfs(self.Jq, self.I) + Bq*self.Bbhfs(self.Jq,self.I)
                values = eigh(H)[0].real
                indices = np.concatenate(([0], np.cumsum(2*self.Fq[:-1]+1))).astype(int)
                if Aq < 0:
                    self.wq = np.flip(values)[indices]
                else:
                    self.wq = values[indices]
            else:
                self.wq = np.array([0])

            if self.number_of_states == 4:
                Ab = df[df['State'].str.contains(self.dressing_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
                Bb = df[df['State'].str.contains(self.dressing_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6
                if len(self.Fb)>1:
                    H = Ab * self.Hhfs(self.Jb, self.I) + Bb*self.Bbhfs(self.Jb,self.I)
                    values = eigh(H)[0].real
                    indices = np.concatenate(([0], np.cumsum(2*self.Fb[:-1]+1))).astype(int)
                    if Ab < 0:
                        self.wb = np.flip(values)[indices]
                    else:
                        self.wb = values[indices]
                else:
                    self.wb = np.array([0])

            if self.number_of_states == 5:
                if 'dressing2' in self.config["states"].keys(): # dressing with two states
                    Ab = df[df['State'].str.contains(self.dressing_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
                    Bb = df[df['State'].str.contains(self.dressing_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6
                    Ab2 = df[df['State'].str.contains(self.dressing_state2)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
                    Bb2 = df[df['State'].str.contains(self.dressing_state2)]['Hyperfine constant (B)'].to_numpy()[0]*1e6
                    if len(self.Fb)>1:
                        H = Ab * self.Hhfs(self.Jb, self.I) + Bb*self.Bbhfs(self.Jb,self.I)
                        values = eigh(H)[0].real
                        indices = np.concatenate(([0], np.cumsum(2*self.Fb[:-1]+1))).astype(int)
                        if Ab < 0:
                            self.wb = np.flip(values)[indices]
                        else:
                            self.wb = values[indices]
                    else:
                        self.wb = np.array([0])

                    if len(self.Fb2)>1:
                        H = Ab2 * self.Hhfs(self.Jb2, self.I) + Bb2*self.Bbhfs(self.Jb2,self.I)
                        values = eigh(H)[0].real
                        indices = np.concatenate(([0], np.cumsum(2*self.Fb2[:-1]+1))).astype(int)
                        if Ab2 < 0:
                            self.wb2 = np.flip(values)[indices]
                        else:
                            self.wb2 = values[indices]
                    else:
                        self.wb2 = np.array([0])
                        
                    self.dress_state_splitting = (self.angular_frequencies[2] - self.angular_frequencies[3])/(2*np.pi)
                else:
                    # ground state mapping
                    Ab = Ag
                    Bb = Bg
                    if len(self.Fb)>1:
                        H = Ab * self.Hhfs(self.Jb, self.I) + Bb*self.Bbhfs(self.Jb,self.I)
                        values = eigh(H)[0].real
                        indices = np.concatenate(([0], np.cumsum(2*self.Fb[:-1]+1))).astype(int)
                        if Ab < 0:
                            self.wb = np.flip(values)[indices]
                        else:
                            self.wb = values[indices]
                    else:
                        self.wb = np.array([0])
            if self.number_of_states == 6:
                # ground state mapping
                Ab = Ag
                Bb = Bg
                if len(self.Fb)>1:
                    H = Ab * self.Hhfs(self.Jb, self.I) + Bb*self.Bbhfs(self.Jb,self.I)
                    values = eigh(H)[0].real
                    indices = np.concatenate(([0], np.cumsum(2*self.Fb[:-1]+1))).astype(int)
                    if Ab < 0:
                        self.wb = np.flip(values)[indices]
                    else:
                        self.wb = values[indices]
                else:
                    self.wb = np.array([0])

                Ad = df[df['State'].str.contains(self.dressing_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
                Bd = df[df['State'].str.contains(self.dressing_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6
                if len(self.Fd)>1:
                    H = Ad * self.Hhfs(self.Jd, self.I) + Bd*self.Bbhfs(self.Jd,self.I)
                    values = eigh(H)[0].real
                    indices = np.concatenate(([0], np.cumsum(2*self.Fd[:-1]+1))).astype(int)
                    if Ad < 0:
                        self.wd = np.flip(values)[indices]
                    else:
                        self.wd = values[indices]
                else:
                    self.wd = np.array([0])
            
    
    def Wigner6jPrefactorSteck(self, Fdash, J, I):
        return pow(-1, -Fdash+J+1+I) * np.sqrt(( 2*Fdash + 1 ) * (2*J + 1) )

    def transition_strength(self, I, J, F, mF, Jdash, Fdash, mFdash, q): #in terms of reduced dipole moment (J)
        element = ( float(CG(Fdash, mFdash, 1, -q, F, mF).doit())
                *  float( Wigner6j(J, Jdash, 1, Fdash, F, I).doit() ) * self.Wigner6jPrefactorSteck(Fdash, J, I) )
        return element
    
    def coupling(self, J, F, mF, Jdash, Fdash, mFdash):
        """Makes couplnig tensor"""
        Q = np.array([-1, 1])
        coupling_tensor = np.zeros((len(F), len(mF), len(Fdash), len(mFdash), len(Q)))
        if self.splitting == False:
            coupling_tensor[:, :, :, :, 0] = 1
        else:
            #transition_strength function uses sympy which I have't been able to vectorise
            for Fi, F_ in enumerate(F):
                for mFi, mF_ in enumerate(mF):
                    for Fdashi, Fdash_ in enumerate(Fdash):
                        for mFdashi, mFdash_ in enumerate(mFdash):
                            for Qi, Q_ in enumerate(Q):
                                coupling_tensor[Fi, mFi, Fdashi, mFdashi, Qi] = self.transition_strength(self.I, J, F_, mF_, Jdash, Fdash_, mFdash_, Q_)

        return coupling_tensor
    
    def rabi_frequency_to_power(self, Omega, r, index=1):
        # index := which dipole strength to use 
        # returns power in W
        # Rabi frequency in simulation defined as half of usual rabi frequency, this is why *2 rather than /2.
        P = self.rabi_frequency_to_intensity(Omega, index) * np.pi* pow(r, 2)
        return P
    
    def power_to_rabi_frequency(self, P, r, index=1):
        # index := which dipole strength to use 
        # Rabi frequency in simulation defined as half of usual rabi frequency.
        I = P/(np.pi*pow(r,2))
        Omega = self.intensity_to_rabi_frequency(I, index)
        return Omega
    
    def rabi_frequency_to_intensity(self, Omega, index=1):
        # index := which dipole strength to use 
        # Rabi frequency in simulation defined as half of usual rabi frequency, this is why *2 rather than /2.
        I = 2*c*epsilon_0*pow(hbar,2) * pow(Omega/self.reduced_dipoles[index], 2)
        return I
    
    def intensity_to_rabi_frequency(self, I, index=1):
        # index := which dipole strength to use 
        # Rabi frequency in simulation defined as half of usual rabi frequency.
        Omega = self.reduced_dipoles[index]/(hbar) * np.sqrt(I/(2*c*epsilon_0))
        return Omega
    
    def control_pulse_to_energy(self, Control, t, r, index=1):
        """ Takes list representing Control(t), returns energy contained within pulse in Joules"""
        # Control could have two polarisations
        # Control must be in rabi frequency (Hz)
        # t must be in seconds
        # Rabi frequency in simulation defined as half of usual rabi frequency, this is why *2 rather than /2.
        E = np.trapz(2*c*epsilon_0*pow(hbar,2) * pow(Control/self.reduced_dipoles[index], 2), x=t, axis=0) * np.pi * pow(r, 2)
        return E
    
    def set_energy_of_control_pulse(self, E, Control, t, r, index=1):
        """ Sets the amplitude of a Control pulse (temporal shape given by Control over time axis t) such that the energy of the pulse is
            equal to E. """
        E0 = np.trapz(2*c*epsilon_0*pow(hbar,2) * pow(Control/self.reduced_dipoles[index], 2), x=t, axis=0) * np.pi * pow(r, 2) # initial energy of supplied control pulse
        norm = E/E0
        Control_new = Control*np.sqrt(norm)
        return Control_new


    def pV(self, T):
        """ Vapour pressure as a function of temperature (K)"""
        T = np.array(T)
        return np.where(T > self.T_melt, 
                        pow(10, self.Al + self.Bl/T + self.Cl*T + self.Dl*np.log10(T)), 
                        pow(10, self.As + self.Bs/T + self.Cs*T + self.Ds*np.log10(T)))
        
    def Nv(self, T):
        """ Number density as a function of temperature (K) """
        return 133.323*self.pV(T)/(Boltzmann*T)
    
    def optical_depth(self, T, L, index=0):
        """Approximate optical depth for a given temperature (K) and cell length (m), using the reduced dipole moment of a transition. 
        This is the optical depth that would be measured if all of the atoms were stationary.
        index : Determines which transition. """
        cross_section = pow(self.reduced_dipoles[index], 2)*self.angular_frequencies[index]/(self.decay_rates[index]*epsilon_0*hbar*c)
        OD = self.Nv(T)*L*cross_section
        return OD
    
    def effective_optical_depth(self, OD, T, index=0):
        """Approximate effective optical depth for a given temperature and cell length, using the reduced dipole moment of a transition.
        This is the optical depth that would be measured on resonance for a thermally boradened ensemble. 
        index : Determines which transition. """
        width = np.sqrt(Boltzmann*T/(self.mass*pow(c, 2)))*self.angular_frequencies[index]
        ODdash = OD*self.decay_rates[index]/(2*width) * np.sqrt(np.pi*np.log(2))
        return ODdash

class Rb87(Alkalis):
    def __init__(self, config):

        self.atom = 'Rb87'
        
        self.I  = 1.5 
        self.mass = 86.909180520*amu
        self.ng = 5

        self.T_melt = 39.31+273.15
        # liquid phase constants
        self.Al = 15.88253
        self.Bl = -4529.635
        self.Cl = 0.00058663
        self.Dl = -2.99138
        # solid phase constants
        self.As = -94.04826
        self.Bs = -1961.258
        self.Cs = -0.03771687
        self.Ds = 42.57526

        self.config = config

        self.filename = DIRECTORY + '/libs/AtomicConstants/Rb/Rb1MatrixElements.csv'
        self.filename_hyperfine =  DIRECTORY + '/libs/AtomicConstants/Rb/Rb87_hyperfine.csv'

        super().__init__()

class Rb85(Alkalis):
    def __init__(self, config):

        self.atom = 'Rb85'
        
        self.I  = 2.5 
        self.mass = 84.9117897*amu
        self.ng = 5

        ### Same as Rb87
        self.T_melt = 39.3+273.15
        # liquid phase constants
        self.Al = 15.88253
        self.Bl = -4529.635
        self.Cl = 0.00058663
        self.Dl = -2.99138
        # solid phase constants
        self.As = -94.04826
        self.Bs = -1961.258
        self.Cs = -0.03771687
        self.Ds = 42.57526

        self.config = config

        self.filename = DIRECTORY + 'libs\\AtomicConstants\\Rb\\Rb1MatrixElements.csv'
        self.filename_hyperfine =  DIRECTORY + 'libs\\AtomicConstants\\Rb\\Rb85_hyperfine.csv'

        super().__init__()

class Cs(Alkalis):
    def __init__(self, config):

        self.atom = 'Cs'
        
        self.I  = 3.5 
        self.mass = 132.905451931*amu
        self.ng = 6

        self.T_melt = 28.44+273.15
        # liquid phase constants
        self.Al = 8.22127
        self.Bl = -4006.048
        self.Cl = - 0.00060194
        self.Dl = -0.19623
        # solid phase constants
        self.As = -219.482
        self.Bs = 1088.676
        self.Cs = -0.08336185
        self.Ds = 94.88752

        self.config = config

        self.filename = DIRECTORY + 'libs\\AtomicConstants\\Cs\\Cs1MatrixElements.csv'
        self.filename_hyperfine =  DIRECTORY + 'libs\\AtomicConstants\\Cs\\Cs_hyperfine.csv'

        super().__init__()

# # K

# # Na

# # Li