import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
from time import time
N = 2
init_random = np.random.random((N,N))
lattice_n = np.zeros((N, N))
lattice_n[init_random>=0.5] = 1
lattice_n[init_random<0.5] = -1

init_random = np.random.random((N,N))
lattice_p = np.zeros((N, N))
lattice_p[init_random>=0.25] = 1
lattice_p[init_random<0.25] = -1
def get_energy(lattice):
    # applies the nearest neighbours summation
    kern = generate_binary_structure(2, 1) 
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern, mode='constant', cval=0)
    return arr.sum()
@numba.njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nopython=True, nogil=True)
def metropolis(spin_arr, times, BJ, energy):
    spin_arr = spin_arr.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        spin_i = spin_arr[x,y] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy
        E_i = 0
        E_f = 0
        if x>0:
            E_i += -spin_i*spin_arr[x-1,y]
            E_f += -spin_f*spin_arr[x-1,y]
        if x<N-1:
            E_i += -spin_i*spin_arr[x+1,y]
            E_f += -spin_f*spin_arr[x+1,y]
        if y>0:
            E_i += -spin_i*spin_arr[x,y-1]
            E_f += -spin_f*spin_arr[x,y-1]
        if y<N-1:
            E_i += -spin_i*spin_arr[x,y+1]
            E_f += -spin_f*spin_arr[x,y+1]
        
        # 3 / 4. change state with designated probabilities
        dE = E_f-E_i
        if (dE>0)*(np.random.random() < np.exp(-BJ*dE)):
            spin_arr[x,y]=spin_f
            energy += dE
        elif dE<=0:
            spin_arr[x,y]=spin_f
            energy += dE
            
        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy
#start = time()
#spins, energies = metropolis(lattice_n, 1000000, 1.75, get_energy(lattice_n))
#end = time()
#print("s",end-start)
#fig, axes = plt.subplots(1, 2, figsize=(12,4))
#ax = axes[0]
#ax.plot(spins/N**2)
#ax.set_xlabel('Algorithm Time Steps')
#ax.set_ylabel(r'Average Spin $\bar{m}$')
#ax.grid()
#ax = axes[1]
#ax.plot(energies)
#ax.set_xlabel('Algorithm Time Steps')
#ax.set_ylabel(r'Energy $E/J$')
#ax.grid()
#fig.tight_layout()
#fig.suptitle(r'Evolution of Average Spin and Energy for $\beta J=$0.7', y=1.07, size=18)
#plt.show()
def auto_time(auto):
    for i in range(len(auto)):
        if np.abs(auto[i]) < 0.07:
            zeroindex = i
            break
    time = np.sum(auto[:zeroindex])
    return time ,zeroindex

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_energy(lattice):
    convolution_mask = generate_binary_structure(2,1)
    convolution_mask[1,1] = False
    Energy =  -lattice*convolve(lattice,convolution_mask, mode = "wrap")/2
    Energy = Energy.sum()
    return Energy

def get_lattice(L,spinRatio = 0.5):
    random = np.random.random((L,L))
    lattice = np.zeros((L,L))
    lattice[random>=spinRatio] = 1
    lattice[random<spinRatio] = -1
    return lattice

@njit( nopython=True, nogil=True)
def fastsweep(lattice ,beta, L, startEnergy):
    all_Energies = np.zeros(L)
    all_Magnets = np.zeros(L)
    all_Energies[0] = startEnergy
    all_Magnets[0] = lattice.sum()
    for t in range(1,L):
        E_p = 0
        E_t = 0
        old_lattice = lattice.copy()
        i = np.random.randint(0,len(lattice)) 
        j = np.random.randint(0,len(lattice))
        flipped_spin = lattice[i,j] 
        lattice[i,j] = flipped_spin*(-1)
        for n in range(2):
            E_t += lattice[i,(j+(-1)**n)%len(lattice)]
            E_t += lattice[(i+(-1)**n)%len(lattice),j]
            E_p += old_lattice[i,(j+(-1)**n)%len(lattice)]
            E_p += old_lattice[(i+(-1)**n)%len(lattice),j]

        E_t = -E_t*lattice[i,j]
        E_p = -E_p*flipped_spin
        deltaE = E_t-E_p
        if np.exp(-beta*(deltaE)) < np.random.random():
            lattice = old_lattice
            all_Energies[t] = all_Energies[t-1]
            all_Magnets[t] = all_Magnets[t-1]
        else:
            all_Energies[t] = deltaE + all_Energies[t-1]
            all_Magnets[t]  = lattice.sum()
        
    return all_Energies,all_Magnets, lattice

lattice = get_lattice(8)
energy = get_energy(lattice)
e, m , lattice = fastsweep(lattice,0.6,1000000,energy)
plt.plot(e)
plt.show()
print(get_energy(lattice)/len(lattice)**2)
print(e[-1]/(len(lattice)**2))
M,E = metropolis(lattice,1000000,0.2,energy)
plt.plot(e)
plt.show()