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
import pygame

__author__ = "Aleksey Sokolov"
#12004091

path = os.getcwd() + "\\A3\\graphics\\"
#my implementation
@njit("UniTuple(f8[:], 2)(f8[:,:], f8, i8, f8)", nopython=True, nogil=True)
def fastsweep(lattice ,beta, L, startEnergy):
    all_Energies = np.zeros(L)
    all_Magnets = np.zeros(L)
    all_Energies_avg = np.zeros(L)
    all_Magnets_avg = np.zeros(L)
    all_Energies[0] = startEnergy
    all_Magnets[0] = lattice.sum()
    loadingbar = int(L/10)
    for t in range(1,L):
        if t%(loadingbar) == 0:
            print(t/L *100," %")
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
          

        E_n = -E_t*lattice[i,j]
        E_p = -E_t*flipped_spin
        deltaE = E_n-E_p
        if np.exp(-beta*deltaE) > np.random.random():
            all_Energies[t] = deltaE + all_Energies[t-1]
            all_Magnets[t]  = lattice.sum()
        else:
            lattice = old_lattice
            all_Energies[t] = all_Energies[t-1]
            all_Magnets[t] = all_Magnets[t-1]
       
    return all_Energies,all_Magnets

#for game (i think the intended sweep implimentation)
@njit
def fastsweepgame(lattice ,beta, startEnergy):
    energy = startEnergy
    for t in range(1,len(lattice)**2):
        
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
          

        E_n = -E_t*lattice[i,j]
        E_p = -E_t*flipped_spin
        deltaE = E_n-E_p
        if np.exp(-beta*deltaE) > np.random.random():
            energy = deltaE + energy
        else:
            lattice = old_lattice
      
    return lattice, energy
@njit("UniTuple(f8[:], 2)(f8[:,:], f8[:])", nopython=True, nogil=True)
def convertToSweeps(lattice, Mchain):
    N = len(lattice)**2
    sweepchain = Mchain[0::N]/N
    runningAvg = np.zeros(len(sweepchain) - 1, dtype=np.float64)
    
    current_sum = 0.0
    for index in range(1, len(sweepchain)):
        current_sum += sweepchain[index - 1]
        runningAvg[index - 1] = current_sum / index
        
        
    return sweepchain, runningAvg

def theory_energy(x):
    A = np.exp(8*x) - np.exp(-8*x)
    B = np.exp(8*x) + np.exp(-8*x) + 6
    return -2*(A/B)

def error_estimate(auto_time, Mchain):
    return np.sqrt((2*auto_time*np.std(Mchain)**2)/len(Mchain))
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val):
        self.rect = pygame.Rect(x, y, w, h)
        self.critrect = pygame.Rect(x+w*0.44069,y,3,h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.dragging = False
        self.font  = pygame.font.SysFont("Times New Roman", 18)
        
         

    def draw(self, screen):
       
        pygame.draw.rect(screen, (200,200,200), self.rect)
        pygame.draw.rect(screen,(255,0,0),self.critrect)
        handle_pos = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.w
        pygame.draw.circle(screen, (0,255,0), (int(handle_pos), self.rect.centery), self.rect.h // 2)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                # Update the slider value
                rel_x = event.pos[0] - self.rect.x
                self.value = self.min_val + (rel_x / self.rect.w) * (self.max_val - self.min_val)
                self.value = max(self.min_val, min(self.value, self.max_val))
    def show_beta(self,screen,x,y):
        beta = self.value
        betarect = pygame.Rect(x,y,60,20)
        pygame.draw.rect(screen,(200,200,200),betarect)
        beta_gui = self.font.render(str(round(beta,3)),1,(255,0,0))
        screen.blit(beta_gui,(x+5,y))

        
        



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


def main(a2b5 = False, a3 = False,bc3 = False, a4 = False, b4 = False,a5 = False):

    if a2b5:
        L_all = [2,4,8,16,32]
        num_of_sweeps = 100000
        for L in L_all:
            BetaJ = np.arange(0.1,1,0.01)
            Avg_Energies = np.zeros(len(BetaJ))
            Avg_Magnets = np.zeros(len(BetaJ))
           
            start = time()
            for j, Bj in tqdm(enumerate(BetaJ), "a2 loop", total=len(BetaJ)):
                lattice = get_lattice(L,0.5)
                lattice_energy = get_energy(lattice)
                Energy, Magnets = fastsweep(lattice,Bj,num_of_sweeps*L**2,lattice_energy)
                Avg_Energies[j] = np.mean(Energy[int(len(Energy)*0.2):]/len(lattice)**2)
                Avg_Magnets[j] = np.mean(np.abs(Magnets[int(len(Energy)*0.2):]/len(lattice)**2))
            end = time()
            runtime = end - start
            plt.style.use('bmh')
            cmap = get_cmap(50,"winter")
            figa2, axa2 = plt.subplots(2, figsize= (16,9))
            plt.suptitle("$\\Delta J \\beta$ = 0.01 coldstart $90\\%$ spin up")
            axa2[0].plot(BetaJ, Avg_Energies, label = f"Average Energies {L}x{L} Lattice \n Tmax = 2e6",color = cmap(20))
            if L == 2:
                axa2[0].plot(BetaJ, theory_energy(BetaJ), "r--",label = "Energy Estimator: \n $-2 \\frac{e^{8 \\beta J}- e^{-8 \\beta J}}{e^{8 \\beta J}+ e^{-8 \\beta J} + 6}$")
            axa2[0].scatter([BetaJ[30]], [Avg_Energies[30]],s = 50, marker="+", color = "r", label =f"$\\beta J$ = 0.4 \n E = {Avg_Energies[30]:.3f}")
            axa2[1].vlines([0.44068],0,1, linestyles= "dashed", colors = "grey", label = f"$J \\beta_c$ = {0.44068}")
            #axa2[0].hlines([Avg_Energies[3]],0.1,1.75, linestyles= "dashed", colors = "grey")
            axa2[0].set_xlabel("$\\beta J$")
            axa2[0].set_ylabel("$\\bar{E}$")
            axa2[0].legend()
            axa2[1].plot(BetaJ, Avg_Magnets, label = f"Average Magnetization {L}x{L}  Lattice \n Tmax = 2e6",color = cmap(40))
            axa2[1].set_xlabel("$\\beta J$")
            axa2[1].set_ylabel("$\\bar{M}$")
            axa2[1].text(0.9,0.1,f"Runtime: {runtime/60:.0f} min {runtime%60:.1f} s",bbox=dict(facecolor=cmap(40), alpha=0.5))
            axa2[1].legend()
            aufgabe = 2
            if L > 2:
                aufgabe = 5
            figa2.savefig(path+ f"{aufgabe}a_{L}x{L}LatticeAvgEnergyMagnetoverT_vs_theory.pdf")
    if a3:
        BetaJ = [0.1,0.4,0.9]
        L_array = [300,300,10000]
        for Bj, L  in zip(BetaJ,L_array):
            lattice = get_lattice(2)
            lattice_energy = get_energy(lattice)
            Energy, Magnets = sweep(lattice,Bj,L,lattice_energy, sequencelength=len(lattice)**2)
    
    if bc3:
        BetaJ = np.arange(0.1,0.7,0.1)
        num_of_sweeps = 100000
        plt.style.use('bmh')
        cmap = get_cmap(50,"winter")
        figb3, axb3 = plt.subplots(3,2, figsize= (16,16))
        for j,BJ in tqdm(enumerate(BetaJ),"b3 loop",total=len(BetaJ)):
            lattice = get_lattice(2)
            E0 = get_energy(lattice)
            Energy , Magnet = fastsweep(lattice,BJ,num_of_sweeps*4,E0)
            
            E_Auto = acf(Energy,nlags=1000000)
            M_Auto = acf(Magnet,nlags=1000000)
            E_auto_time, zero_E = auto_time(E_Auto)
            M_auto_time, zero_M = auto_time(M_Auto)
            error_E_mean =  (2*E_auto_time*np.std(Energy/len(lattice)**2)**2)/len(Energy)
            error_M_mean =  (2*M_auto_time*np.std(Magnet/len(lattice)**2)**2)/len(Magnet)
            axb3_2 = axb3[j%3,j//3].twinx()
            axb3[j%3,j//3].plot(np.log(np.arange(0,len(E_Auto),1)), E_Auto, label=f"Autocorr E \n $\\beta J$ = {BJ:.2f} \n $\\tau$ = {E_auto_time:.2f} \n" + "$\\bar{E}$=" + f" {np.mean(Energy)/len(lattice)**2:.3f} $\pm$ {np.sqrt(error_E_mean):.3f}", color= cmap(7))
            axb3[j%3,j//3].set_ylabel("$\\rho_{E}$", color= cmap(7))
            axb3[j%3,j//3].spines['left'].set_color(cmap(7))
            axb3[j%3,j//3].tick_params(axis='y', colors=cmap(7))
            axb3_2.plot(np.log(np.arange(0,len(M_Auto),1)), M_Auto, label=f"Autocorr M \n $\\beta J$ = {BJ:.2f} \n $\\tau$ = {M_auto_time:.2f} \n"+ "$\\bar{M}$=" + f" {np.mean(Magnet)/len(lattice)**2:.3f} $\pm$ {np.sqrt(error_M_mean):.3f}",color= cmap(43))
            axb3_2.set_ylabel("$\\rho_{M}$", color= cmap(43))
            axb3_2.spines['right'].set_color(cmap(43))
            axb3_2.tick_params(axis='y', colors=cmap(43))
            axb3_2.legend(loc="right")
            axb3[j%3,j//3].legend(loc= "upper right")
        figb3.savefig(path + "3bc_Autocorr2x2_different_BJ.pdf")
    if a4:
        BetaJ = [0.2,0.4,0.6]
        L_array = [4,6,8]
        num_of_sweeps = 1000000
        plt.style.use('bmh')
        cmap = get_cmap(50,"winter")
        figa4, axa4 = plt.subplots(3,3, figsize= (16,16))
        #axa4_em.set_ylim(min(Magnet)-1,np.abs(min(Energy)))
        figa4.tight_layout(pad=5)
        for i, L in tqdm(enumerate(L_array),"L loop a4", total = len(L_array)):
            figa4_em, axa4_em = plt.subplots(6, figsize= (18,16))
            figa4_em.suptitle(f"{L}x{L} Lattice")
            figa4_em.tight_layout(pad=5.0)
            k = 0
            for j,BJ in tqdm(enumerate(BetaJ),"beta a4 loop",total=len(BetaJ)):
                lattice = get_lattice(L)
                E0 = get_energy(lattice)
                Energy , Magnet = fastsweep(lattice,BJ,L*L*num_of_sweeps,E0)
                if (L == 6) or (L == 8):
                    Energy = Energy[int(len(Energy)*0.2):]
                    Magnet = Magnet[int(len(Magnet)*0.2):]
                Magnet, Magnet_avg  = convertToSweeps(lattice,Magnet)
                Energy, Energy_avg = convertToSweeps(lattice,Energy)
                E_Auto = acf(Energy,nlags=L*L*num_of_sweeps)
                M_Auto = acf(Magnet,nlags=L*L*num_of_sweeps)
                E_auto_time, zero_E = auto_time(E_Auto)
                M_auto_time, zero_M = auto_time(M_Auto)
                error_E_mean =  (2*E_auto_time*np.std(Energy)**2)/len(Energy)
                error_M_mean =  (2*M_auto_time*np.std(Magnet)**2)/len(Magnet)

                axa4_2 = axa4[i,j].twinx()
                axa4[i,j].plot(np.log(np.arange(0.1,len(E_Auto)+0.1,1)), E_Auto, label=f"Autocorr E \n $\\beta J$ = {BJ:.2f} \n $\\tau$ = {E_auto_time:.3f} \n" + "$\\bar{E}$=" + f" {np.mean(Energy):.3f} $\pm$ {np.sqrt(error_E_mean):.3f} \n Size = {L}X{L}", color= cmap(7))
                axa4[i,j].set_ylabel("$\\rho_{E}$", color= cmap(7))
                axa4[i,j].spines['left'].set_color(cmap(7))
                axa4[i,j].tick_params(axis='y', colors=cmap(7))
                
                axa4_2.plot(np.log(np.arange(0.1,len(M_Auto)+0.1,1)), M_Auto, label=f"Autocorr M \n $\\beta J$ = {BJ:.2f} \n $\\tau$ = {M_auto_time:.3f} \n"+ "$\\bar{M}$=" + f" {np.mean(Magnet):.3f} $\pm$ {np.sqrt(error_M_mean):.3f} \n Size = {L}X{L}",color= cmap(43))
                axa4_2.set_ylabel("$\\rho_{M}$", color= cmap(43))
                axa4_2.spines['right'].set_color(cmap(43))
                axa4_2.tick_params(axis='y', colors=cmap(43))
                axa4_2.legend(loc="right")
                axa4[i,j].legend(loc= "upper right")
                #timeseries
                axa4_em[j+k].title.set_text( f" $\\beta J$ = {BJ}")
                axa4_em[j+k].grid()
                axa4_em[j+k].set_ylim(-3,1)
                axa4_em[j+k].plot(np.arange(0,len(Energy),1), Energy, label=f"Energy per site \n $\\beta J$ = {BJ:.2f} \n" + "$\\bar{E}$=" + f" {np.mean(Energy):.3f} $\pm$ {np.sqrt(error_E_mean):.3f} \n Size = {L}X{L}", color= cmap(7))
                axa4_em[j+k].plot(np.arange(0,len(Energy)-1,1),Energy_avg,"r-", label = "Energy Running average")
                axa4_em[j+k].set_ylabel("$E$ / Energy", color= cmap(7))
                axa4_em[j+k].spines['left'].set_color(cmap(7))
                axa4_em[j+k].tick_params(axis='y', colors=cmap(7))

                axa4_em[j+k+1].set_ylim(-2,2)
                axa4_em[j+k+1].grid()
                axa4_em[j+k+1].plot(np.arange(0,len(Magnet),1), Magnet, label=f"Magnetization \n $\\beta J$ = {BJ:.2f} \n"+ "$\\bar{M}$=" + f" {np.mean(Magnet):.3f} $\pm$ {np.sqrt(error_M_mean):.3f} \n Size = {L}X{L}",color= cmap(43))
                axa4_em[j+k+1].plot(np.arange(0,len(Magnet)-1,1),Magnet_avg, "r-", label = "Magnetization Running average")
                axa4_em[j+k+1].set_ylabel("$M$ / Magnetization", color= cmap(43))
                axa4_em[j+k+1].spines['right'].set_color(cmap(43))
                axa4_em[j+k+1].tick_params(axis='y', colors=cmap(43))
                axa4_em[j+k+1].legend(loc="right")
                axa4_em[j+k].legend(loc= "upper right")
                k += 1
            figa4_em.savefig(path + f"4a_Energy_magnet_{L}x{L}_lattice.pdf")
            figa4_em.clf()
        figa4.savefig(path + "4a_autocorr_multiple_lattices.pdf")

    if b4:
        BetaJ = 0.44069
        L_all = [4,8,16,32]
        num_of_sweeps = 10000
        fig_b4 , ax_b4 = plt.subplots(4,3, figsize= (24,16),gridspec_kw={'width_ratios': [2,2, 1]})
        fig_b4.suptitle(f"$\\beta J = \\beta_c J = 0.44069$ sweeps = {num_of_sweeps}")
        k = 0
        plt.style.use('bmh')
        cmap = get_cmap(50,"winter")
        for i, L in tqdm(enumerate(L_all), desc= "b4 loop",total=len(L_all)):
            lattice = get_lattice(L)
            E0 = get_energy(lattice)
            Energy, Magnets = fastsweep(lattice,BetaJ,num_of_sweeps*L**2,E0)
            Magnet, Magnet_avg  = convertToSweeps(lattice,Magnets)
            Energy, Energy_avg = convertToSweeps(lattice,Energy)
            E_Auto = acf(Energy,nlags=L*L*num_of_sweeps)
            M_Auto = acf(Magnet,nlags=L*L*num_of_sweeps)
            #ax_b4[i,0].plot()
            E_Auto_time,_ = auto_time(E_Auto)
            M_Auto_time,_ = auto_time(M_Auto)
            error_E_mean =  error_estimate(E_Auto_time,Energy)
            error_M_mean =  error_estimate(M_Auto_time,Magnet)

            print("E_error",error_E_mean)
            print("E std", np.std(Energy))
            print("E auto", E_Auto_time)
            print("M_error",error_M_mean)
            print("E std", np.std(Magnet))
            print("M auto", M_Auto_time)

            ax_b4[i,0].title.set_text( f" {L}x{L} Lattice")
            ax_b4[i,0].grid()
            ax_b4[i,0].set_ylim(-3,1)
            ax_b4[i,0].plot(np.arange(0,len(Energy),1), Energy, label="Energy per site \n" + "$\\bar{E}$=" + f" {np.mean(Energy):.3f} $\pm$ {error_E_mean:.3f} \n Size = {L}X{L}", color= cmap(7))
            ax_b4[i,0].plot(np.arange(0,len(Energy)-1,1),Energy_avg,"r-", label = "Energy Running average")
            ax_b4[i,0].set_ylabel("$E$ / Energy", color= cmap(7))
            ax_b4[i,0].spines['left'].set_color(cmap(7))
            ax_b4[i,0].tick_params(axis='y', colors=cmap(7))
            ax_b4[i,1].title.set_text( f" {L}x{L} Lattice")
            ax_b4[i,1].set_ylim(-2,2)
            ax_b4[i,1].grid()
            ax_b4[i,1].plot(np.arange(0,len(Magnet),1), Magnet, label=f"Magnetization \n "+ "$\\bar{M}$=" + f" {np.mean(Magnet):.3f} $\pm$ {error_M_mean:.3f} \n Size = {L}X{L}",color= cmap(43))
            ax_b4[i,1].plot(np.arange(0,len(Magnet)-1,1),Magnet_avg, "r-", label = "Magnetization Running average")
            ax_b4[i,1].set_ylabel("$M$ / Magnetization", color= cmap(43))
            ax_b4[i,1].spines['right'].set_color(cmap(43))
            ax_b4[i,1].tick_params(axis='y', colors=cmap(43))
            ax_b4[i,1].legend(loc="upper right")
            ax_b4[i,0].legend(loc= "upper right")
            ax_b4_2 = ax_b4[i,2].twinx()
            ax_b4[i,2].title.set_text( f" {L}x{L} Lattice")
            ax_b4[i,2].plot(np.log(np.arange(0.1,len(E_Auto)+0.1,1)), E_Auto, label=f"Autocorr E \n  $\\tau$ = {E_Auto_time:.2f} \n" + "$\\bar{E}$=" + f" {np.mean(Energy):.3f} $\pm$ {error_E_mean:.3f} \n Size = {L}X{L}", color= cmap(7))
            ax_b4[i,2].set_ylabel("$\\rho_{E}$", color= cmap(7))
            ax_b4[i,2].spines['left'].set_color(cmap(7))
            ax_b4[i,2].tick_params(axis='y', colors=cmap(7))
            ax_b4_2.plot(np.log(np.arange(0.1,len(M_Auto)+0.1,1)), M_Auto, label=f"Autocorr M \n  $\\tau$ = {M_Auto_time:.2f} \n"+ "$\\bar{M}$=" + f" {np.mean(Magnet):.3f} $\pm$ {error_M_mean:.3f} \n Size = {L}X{L}",color= cmap(43))
            ax_b4_2.set_ylabel("$\\rho_{M}$", color= cmap(43))
            ax_b4_2.spines['right'].set_color(cmap(43))
            ax_b4_2.tick_params(axis='y', colors=cmap(43))
            ax_b4_2.legend(loc="upper right")
            ax_b4[i,2].legend(loc= "right")
            k += 1
        fig_b4.savefig(path + f"4b_betac_all_lattices{num_of_sweeps}sweeps.pdf")
    if a5:
        beta = 0.7
        pygame.init()

        width, height = 800, 800
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Ising Model')
        lattice = get_lattice(100)
        energy = get_energy(lattice)
        rows, cols = lattice.shape[0], lattice.shape[1]  
        cell_width = width // cols
        cell_height = height // rows

        norm_lattice = np.zeros((len(lattice),len(lattice)))
        norm_lattice[lattice<0] = 0
    
        def value_to_color(value):
            gray = int(value * 255)
            return (gray, gray, gray)
        slider = Slider(100,700,600,40,0.01,1,0.5)
   
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                slider.handle_event(event)

            beta = slider.value
            screen.fill((0, 0, 0))

            for i in range(rows):
                for j in range(cols):
                    color = value_to_color(norm_lattice[i][j])
                    pygame.draw.rect(
                        screen,
                        color,
                        [j * cell_width, i * cell_height, cell_width, cell_height]
                    )
            slider.draw(screen)
            slider.show_beta(screen,370,750)
            
            pygame.display.flip()

            clock.tick(30)
            lattice , energy = fastsweepgame(lattice,beta,energy)
            norm_lattice = (lattice - np.min(lattice)) / (np.max(lattice) - np.min(lattice))

        pygame.quit()

       
        
   
    return


if __name__ == '__main__':
    main(a2b5=False, a3= False , bc3 = True, a4 =False,b4 = False, a5=False)




































