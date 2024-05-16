import sys
sys.path.append('C:/ComputerSimulations/')
from _numsolvers import methods as meth
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.stattools import acf
from time import time
def sweep_avg(energy,magnet,lattice):
    L = len(lattice)
    chunk_amount = (len(energy)//L**2)
    chunk_size = chunk_amount*L**2
    en_spl = np.mean(np.split(energy[:chunk_size],chunk_amount), axis = 1)
    M_spl = np.mean(np.split(magnet[:chunk_size],chunk_amount), axis = 1)
    return en_spl, M_spl
L = 50
lattice = meth.get_lattice(L,0.5)
plt.imshow(lattice)
plt.show()
energy = meth.get_energy(lattice)
E,m,l = meth.sweep(lattice,10,1000000,energy,sequencelength=len(lattice)**2)
#E,M,lat = meth.sweep(lattice,0.1,1000000*L**2,energy,L**2)
#start = time()
#E, M = meth.fastsweep(lattice,0.9,1000000*L**2,energy)
#end = time()
#runtime = end-start
#print(f"runtime {runtime/60:.0f} min {runtime%60:.2f} s")
#fig, ax = plt.subplots(2,figsize=(16,9))
#ax[0].plot(E, label = "energy")
#ax[0].legend()
#ax[1].plot(M, label = "magnets")
#ax[1].legend()
#fig.savefig("millionsweeps32x32_beta09.pdf")
E,M = sweep_avg(E,M,lattice)
fig2, ax2 = plt.subplots(2,figsize=(16,9))
ax2[0].plot(E, label = "energy")
ax2[0].legend()
ax2[1].plot(M, label = "magnets")
ax2[1].legend()
fig2.savefig("millionsweeps32x32_beta09_avg.pdf")
