import sys
sys.path.append('C:/ComputerSimulations/')
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from time import time
from _numsolvers import methods as meth
import numba 
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
import matplotlib.style as mplstyle
import matplotlib as mpl
from statsmodels.tsa.stattools import acf
path = os.getcwd() + "\\A3\\graphics\\"
@njit("UniTuple(f8[:], 2)(f8[:,:], f8, i8, f8)", nopython=True, nogil=True)
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
        
    return all_Energies,all_Magnets
#@njit("float64[:](float64[:])" ,nopython=True, nogil=True)
def auto_corr_func(x):
    x_0 = x - np.mean(x)
    cov = np.zeros(len(x))
    cov[0] = x_0.dot(x_0) #variance of the data with itself
    #equivalent to a convolution kind of
    for i in tqdm(range(len(x)-1),"auto corr loop "):
            cov[i + 1] = x_0[i + 1 :].dot(x_0[: -(i + 1)])
    return cov/cov[0] 


def main(a2 = False, a3 = False,bc3 = False, a4 = False):

    if a2:
        BetaJ = np.arange(0.1,2,0.1)
        Avg_Energies = np.zeros(len(BetaJ))
        Avg_Magnets = np.zeros(len(BetaJ))
        for j, Bj in tqdm(enumerate(BetaJ), "a2 loop"):
            lattice = get_lattice(2)
            lattice_energy = get_energy(lattice)
            Energy, Magnets = fastsweep(lattice,Bj,1000000,lattice_energy)
            Avg_Energies[j] = np.mean(Energy)/len(lattice)**2
            Avg_Magnets[j] = np.mean(Magnets)/len(lattice)**2

        plt.style.use('bmh')
        cmap = get_cmap(50,"winter")
        figa2, axa2 = plt.subplots(2, figsize= (16,9))

        axa2[0].plot(BetaJ, Avg_Energies, label = "Average Energies 2x2 Lattice \n Tmax = 1e6",color = cmap(20))
        axa2[0].scatter([BetaJ[3]], [Avg_Energies[3]],s = 50, marker="+", color = "r", label =f"$\\beta J$ = 0.4 \n E = {Avg_Energies[3]:.3f}")
        #axa2[0].vlines([BetaJ[3]],-2,-0.4, linestyles= "dashed", colors = "grey")
        #axa2[0].hlines([Avg_Energies[3]],0.1,1.75, linestyles= "dashed", colors = "grey")
        axa2[0].set_xlabel("$\\beta J$")
        axa2[0].set_ylabel("$\\bar{E}$")
        axa2[0].legend()
        axa2[1].plot(BetaJ, Avg_Magnets, label = "Average Magnetization 2x2 Lattice \n Tmax = 1e6",color = cmap(40))
        axa2[1].set_xlabel("$\\beta J$")
        axa2[1].set_ylabel("$\\bar{M}$")
        axa2[1].legend()
        figa2.savefig(path+ "2x2LatticeAvg3.pdf")
    if a3:
        BetaJ = [0.1,0.4,0.9]
        L_array = [300,300,10000]
        for Bj, L  in zip(BetaJ,L_array):
            lattice = get_lattice(2)
            lattice_energy = get_energy(lattice)
            Energy, Magnets = sweep(lattice,Bj,L,lattice_energy, sequencelength=L/100)
    
    if bc3:
        BetaJ = np.arange(0.1,0.7,0.1)
        plt.style.use('bmh')
        cmap = get_cmap(50,"winter")
        figb3, axb3 = plt.subplots(3,2, figsize= (16,16))
        for j,BJ in tqdm(enumerate(BetaJ),"b3 loop",total=len(BetaJ)):
            lattice = get_lattice(2)
            E0 = get_energy(lattice)
            Energy , Magnet = fastsweep(lattice,BJ,1000000,E0)
            
            E_Auto = acf(Energy,nlags=1000000)
            M_Auto = acf(Magnet,nlags=1000000)
            E_auto_time, zero_E = auto_time(E_Auto)
            M_auto_time, zero_M = auto_time(M_Auto)
            error_E_mean =  (2*E_auto_time*np.std(Energy[:zero_E]/len(lattice)**2)**2)/len(Energy[:zero_E])
            error_M_mean =  (2*M_auto_time*np.std(Magnet[:zero_M]/len(lattice)**2)**2)/len(Magnet[:zero_M])
            axb3_2 = axb3[j%3,j//3].twinx()
            axb3[j%3,j//3].plot(np.log(np.arange(0,len(E_Auto),1)), E_Auto, label=f"Autocorr E \n $\\beta J$ = {BJ:.2f} \n $\\tau$ = {E_auto_time:.2f} \n" + "$\\bar{E}$=" + f" {np.mean(Energy)/len(lattice)**2:.2f} $\pm$ {np.sqrt(error_E_mean):.2f}", color= cmap(7))
            axb3[j%3,j//3].set_ylabel("$\\rho_{E}$", color= cmap(7))
            axb3[j%3,j//3].spines['left'].set_color(cmap(7))
            axb3[j%3,j//3].tick_params(axis='y', colors=cmap(7))
            axb3_2.plot(np.log(np.arange(0,len(M_Auto),1)), M_Auto, label=f"Autocorr M \n $\\beta J$ = {BJ:.2f} \n $\\tau$ = {M_auto_time:.2f} \n"+ "$\\bar{M}$=" + f" {np.mean(Magnet)/len(lattice)**2:.2f} $\pm$ {np.sqrt(error_M_mean):.2f}",color= cmap(43))
            axb3_2.set_ylabel("$\\rho_{M}$", color= cmap(43))
            axb3_2.spines['right'].set_color(cmap(43))
            axb3_2.tick_params(axis='y', colors=cmap(43))
            axb3_2.legend(loc="right")
            axb3[j%3,j//3].legend(loc= "upper right")
        figb3.savefig(path + "Autocorr2x2_different_BJ.pdf")
    if a4:
        BetaJ = [0.2,0.4,0.6]
        L_array = [4,6,8]
        plt.style.use('bmh')
        cmap = get_cmap(50,"winter")
        figa4, axa4 = plt.subplots(3,3, figsize= (16,16))
        figa4_em, axa4_em = plt.subplots(3,3, figsize= (20,16))
        figa4.tight_layout(pad=3)
        for i, L in tqdm(enumerate(L_array),"L loop a4", total = len(L_array)):

            for j,BJ in tqdm(enumerate(BetaJ),"beta a4 loop",total=len(BetaJ)):
                lattice = get_lattice(L)
                E0 = get_energy(lattice)
                Energy , Magnet = fastsweep(lattice,BJ,1000000,E0)

                E_Auto = acf(Energy,nlags=1000000)
                M_Auto = acf(Magnet,nlags=1000000)
                E_auto_time, zero_E = auto_time(E_Auto)
                M_auto_time, zero_M = auto_time(M_Auto)
                error_E_mean =  (2*E_auto_time*np.std(Energy[:zero_E]/len(lattice)**2)**2)/len(Energy[:zero_E])
                error_M_mean =  (2*M_auto_time*np.std(Magnet[:zero_M]/len(lattice)**2)**2)/len(Magnet[:zero_M])
                axa4_2 = axa4[i,j].twinx()
                axa4[i,j].plot(np.log(np.arange(0.1,len(E_Auto)+0.1,1)), E_Auto, label=f"Autocorr E \n $\\beta J$ = {BJ:.2f} \n $\\tau$ = {E_auto_time:.2f} \n" + "$\\bar{E}$=" + f" {np.mean(Energy)/len(lattice)**2:.2f} $\pm$ {np.sqrt(error_E_mean):.2f} \n Size = {L}X{L}", color= cmap(7))
                axa4[i,j].set_ylabel("$\\rho_{E}$", color= cmap(7))
                axa4[i,j].spines['left'].set_color(cmap(7))
                axa4[i,j].tick_params(axis='y', colors=cmap(7))
                axa4_2.plot(np.log(np.arange(0.1,len(M_Auto)+0.1,1)), M_Auto, label=f"Autocorr M \n $\\beta J$ = {BJ:.2f} \n $\\tau$ = {M_auto_time:.2f} \n"+ "$\\bar{M}$=" + f" {np.mean(Magnet)/len(lattice)**2:.2f} $\pm$ {np.sqrt(error_M_mean):.2f} \n Size = {L}X{L}",color= cmap(43))
                axa4_2.set_ylabel("$\\rho_{M}$", color= cmap(43))
                axa4_2.spines['right'].set_color(cmap(43))
                axa4_2.tick_params(axis='y', colors=cmap(43))
                axa4_2.legend(loc="right")
                axa4[i,j].legend(loc= "upper right")


                axa4_em_2 = axa4_em[i,j].twinx()
                axa4_em[i,j].plot(np.arange(0,len(Energy),1), Energy, label=f"Energy per site \n $\\beta J$ = {BJ:.2f} \n" + "$\\bar{E}$=" + f" {np.mean(Energy)/len(lattice)**2:.2f} $\pm$ {np.sqrt(error_E_mean):.2f} \n Size = {L}X{L}", color= cmap(7))
                axa4_em[i,j].set_ylabel("$E$ / Energy$", color= cmap(7))
                axa4_em[i,j].spines['left'].set_color(cmap(7))
                axa4_em[i,j].tick_params(axis='y', colors=cmap(7))
                axa4_em_2.plot(np.arange(0,len(Magnet),1), Magnet, label=f"Magnetization \n $\\beta J$ = {BJ:.2f} \n"+ "$\\bar{M}$=" + f" {np.mean(Magnet)/len(lattice)**2:.2f} $\pm$ {np.sqrt(error_M_mean):.2f} \n Size = {L}X{L}",color= cmap(43))
                axa4_em_2.set_ylabel("$M$ / Magnetization", color= cmap(43))
                axa4_em_2.spines['right'].set_color(cmap(43))
                axa4_em_2.tick_params(axis='y', colors=cmap(43))
                axa4_em_2.legend(loc="right")
                axa4_em[i,j].legend(loc= "upper right")
            figa4.savefig(path + "4a_autocorr_multiple_lattices.pdf")
            figa4_em.savefig(path + "4a_Energy_magnet_multiple_lattices.pdf")
    lattice = get_lattice(200,0.5)
    plt.imshow(lattice)
    plt.show()
    energy = get_energy(lattice)
    E, M = sweep(lattice,1,1000000,energy,sequencelength=10000)
    return





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

def sweep(lattice ,beta, L, startEnergy, sequencelength = 1000):
    all_Energies = np.zeros(L)
    all_Magnets = np.zeros(L)
    all_Energies[0] = startEnergy
    all_Magnets[0] = lattice.sum()
    mplstyle.use('fast')
    fig, ax = plt.subplots(1,3, figsize= (16,9))
    plt.title(f"$\\beta J$ = {beta}")
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
        if (t%(sequencelength) == 0):
            mpl.rcParams['path.simplify'] = True
            mpl.rcParams['path.simplify_threshold'] = 1.0
            ax[0].imshow(lattice)
            ax[1].plot(np.arange(0,t,1),all_Energies[:t]/len(lattice)**2)
            ax[1].set_xlabel("t / simulation time")
            ax[1].set_ylabel("Energy")
            ax[2].plot(np.arange(0,t,1),all_Magnets[:t]/len(lattice)**2)
            ax[2].set_xlabel("t / simulation time")
            ax[2].set_ylabel("Magnetization")
            plt.show(block=False)
            plt.pause(0.001)
            ax[0].cla()
            ax[1].cla()
            ax[2].cla()
        
            
    return all_Energies,all_Magnets


if __name__ == '__main__':
    main(a2=False, a3= False , bc3 = False, a4 = False)



#lattice = get_lattice(2,0.5)
#print(lattice)
#Energy =  -lattice*convolve(lattice,convolution_mask, mode = "wrap")
#print(Energy)
#Energy = Energy.sum()
#print(Energy)
#Energy = get_energy(lattice)
#start = time()
#E,M = fastsweep(lattice,0.4,1000000,Energy)
#end = time()
#print(end-start)
#print(E/4)
#print(np.mean(E/4))
#for i in range(2):
#    plt.figure(figsize=(20,8))
#    start = time()
#    E,M = sweep(lattice,0.4,1000000,Energy)
#    end = time()
#    print(end-start)
#    plt.plot(E/len(lattice)**2)
#    plt.show()
#plt.plot(M)
#plt.show()
#












































