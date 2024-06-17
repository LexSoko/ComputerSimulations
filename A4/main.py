import matplotlib.pyplot as plt
import pandas as pd
import os 
import random
import numpy as np
from scipy.linalg import norm
import matplotlib.style as mplstyle
import matplotlib as mpl
from tqdm import tqdm

__author__ = "Aleksey Sokolov"
#12004091

pathdata = os.getcwd() + "\\A4\\data\\"
pathgraphics = os.getcwd() + "\\A4\\graphics\\"
def change_pos(array, index1, index2):
    temparray = array
    if index1 > index2:
        tempindex = index1
        index1 = index2
        index2 = tempindex
    subarray = temparray[index1:index2+1]
    subarray = subarray[::-1]
    temparray[index1:index2+1] = subarray
    return temparray

def calculate_energy(array):
    shifted_array = np.roll(array,-1)
    delta = shifted_array - array
    energy = 0
    for d in delta:
        energy += norm(d)
    return energy

def plot_GUI(ax_ts,current_array,all_energies_k, all_energies_avg, all_energies_var):
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = 1.0
    ax_ts[0,0].set_aspect("equal")
    ax_ts[0,0].plot(current_array.T[0],current_array.T[1], "b-")
    ax_ts[0,0].plot([current_array.T[0][-1],current_array.T[0][0]],[current_array.T[1][-1],current_array.T[1][0]], "b-")
    ax_ts[0,0].plot(current_array.T[0],current_array.T[1], "ro")


    ax_ts[0,1].plot(all_energies_k[1:], label = f"Best Energy = {all_energies_k[-1]:.3f}")

    ax_ts[1,0].plot(all_energies_avg[1:], label = "$\\bar{E}_k = \\frac{1}{L} \sum_j E_{j,k}$  (L = n_{steps})\n $\\bar{E}_k$ = " + f"{all_energies_avg[-1]:.3f} $\pm$ {np.sqrt(all_energies_var[-1]):.3f}")
    ax_ts[1,0].fill_between(np.arange(0,len(all_energies_avg[1:]),1),all_energies_avg[1:] - np.sqrt(all_energies_var[1:]),all_energies_avg[1:] + np.sqrt(all_energies_var[1:]),alpha = 0.2)

    ax_ts[1,1].plot(all_energies_var[1:], label = "$(\Delta E)^2$ = " + f"{all_energies_var[-1]:.6f}")

    ax_ts[0,1].set_xlabel("$k$")
    ax_ts[0,1].set_ylabel("Energy / arb.U.")

    ax_ts[1,0].set_xlabel("$k$")
    ax_ts[1,0].set_ylabel("Avg Energy / arb.U.")

    ax_ts[1,1].set_ylabel("Variance / arb.U ")
    ax_ts[1,1].set_xlabel("$k$")

    ax_ts[1,0].legend()
    ax_ts[0,1].legend(loc = "upper right")
    ax_ts[1,1].legend()

def find_best_route(array,Tstart,q,startEnergy,nsteps, figname = "lastfirst", plot = True):
    all_energies = [startEnergy]
    all_energies_k = []
    all_energies_avg = []
    all_energies_var = []
    cities_temp = [array]
    current_E = startEnergy
    current_array = np.copy(array)
    current_T = Tstart
    same = 0
    k = 1
    n = 1
    
    fig_ts , ax_ts = plt.subplots(2,2,figsize=(16,16))
    fig_ts.suptitle(f" $q$ = {q} ," + "$T_{start}$ = " +f"{Tstart} ," + "$n_{steps}$ = " +f"{nsteps}")
    mplstyle.use('fast')
    notconverged = 0
    while notconverged<3:
        a = np.random.randint(0,len(array))
        b = np.random.randint(0,len(array))
        if a > b:
            tempindex = a
            a = b
            b = tempindex
        if a != b:
            if ((a,b) != (0,len(array)-1)):
                e_Kprime = norm(current_array[(a-1)%len(current_array)] -current_array[(b)%len(array)]) + norm(current_array[(a)%len(array)] -current_array[(b+1)%len(array)]) 
                e_K = (norm(current_array[(a-1)%len(current_array)]- current_array[(a)%len(array)]) + norm(current_array[(b)%len(array)]- current_array[(b+1)%len(array)]))
                dE = e_Kprime - e_K   
                
            else:
                dE = 0.0 
        else:
            dE = 0.0

        if np.exp(-dE/current_T) > np.random.random():
            if dE != 0.0:
                current_array = change_pos(current_array,a,b)
            current_E += dE
            all_energies.append(current_E)
            
            
        else:
            all_energies.append(current_E)
            same += 1
        if n%nsteps == 0:
            minimum_Energy = np.min(all_energies[-len(array)**2:-1])
            average_Energy = np.mean(all_energies[-len(array)**2:-1])
            variance_Energy = np.var(all_energies[-len(array)**2:-1])
           
            all_energies_k.append(minimum_Energy)
            all_energies_avg.append(average_Energy)
            all_energies_var.append(variance_Energy)
            cities_temp.append(np.copy(current_array))
    
            if k > 1:
                if len(current_array) -np.sum(np.all(cities_temp[-1]== cities_temp[-2], axis=1)) <= 1:
                    notconverged +=1
                if n>20*len(current_array)*nsteps:
                    notconverged +=1
                
            if plot:
                plot_GUI(ax_ts,current_array,all_energies_k, all_energies_avg, all_energies_var)
                ax_ts[1,1].text(0.9,0.9,"$T_k = T_{start} k^{-q}$ = "+ f"{current_T:.3f}")
                plt.show(block=False)
                plt.pause(0.001)
                if notconverged < 3:
                    ax_ts[0, 0].cla()
                    ax_ts[0, 1].cla()
                    ax_ts[1, 0].cla()
                    ax_ts[1, 1].cla()
            k += 1
            current_T = Tstart*(k**(-q))
       
        n += 1
    if plot != True:
        plot_GUI(ax_ts,current_array,all_energies_k, all_energies_avg, all_energies_var)
    
    fig_ts.savefig(f"{figname}.pdf")
    fig_ts.clf()
        

    return current_array, all_energies




def main(a = False, b =False, c = False, d=False):
    
    if a:
    
        data = np.loadtxt(pathdata + "U4 city-positions.txt", delimiter=";")
        
        enegy = calculate_energy(data)

        new_path,en = find_best_route(data,1,1,enegy,len(data)**2, figname=pathgraphics+"Tstart_1_q_1_U4")
        print(f"best Energy reauched: {en[-1]}" )
        new_new_path , ne_en = find_best_route(np.copy(new_path),0.5,1,en[-1],len(new_path)**2,figname=pathgraphics+"secondrun_Tstart_05_q_1")
        print(f"best Energy reauched: {ne_en[-1]}" )
    if b:
        data2 = np.array(pd.read_csv(pathdata+ "albert3.csv", delimiter=","))
        random.shuffle(data2)
        data = data2

        enegy = calculate_energy(data)

        new_path,en = find_best_route(data,1,1,enegy,len(data)**2, figname=pathgraphics+"albert_Tstart_1_q_1_U4")
        print(f"best Energy reached first run: {en[-1]}" )
        new_new_path , ne_en = find_best_route(np.copy(new_path),0.8,1,en[-1],len(new_path)**2,figname=pathgraphics+"albert_secondrun_Tstart_08_q_1")
        print(f"best Energy reached second run: {ne_en[-1]}" )
        new_new_new_path , new_en = find_best_route(np.copy(new_new_path),0.3,2,ne_en[-1],len(new_path)**2,figname=pathgraphics+"albert_thirdrun_Tstart_03_q_2")
        print(f"best Energy reached third run: {new_en[-1]}" )

    if c: 
        
        data = np.loadtxt(pathdata + "U4 city-positions.txt", delimiter=";")
        data2 = data
        enegy = calculate_energy(data2)
       
        nsteps = [len(data), len(data)**2, 2*len(data)**2]
        qsteps = [0.3,0.5,1,3]
        for n in tqdm(nsteps, desc= "nsteps loop"):
            for q in  tqdm(qsteps,desc= "q loop"):
                new_new_path ,en = find_best_route(data,1,q,enegy,n, figname=pathgraphics+f"Different_configs_nsteps_{n}_q_{q}",plot=False)

if __name__ == '__main__':
    main(a= True,b = True,c=True)