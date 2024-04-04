import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter, FFMpegWriter
from tqdm import tqdm
from scipy.stats import norm
import random as rd
from matplotlib import pyplot, transforms
import os
import statsmodels.api as sm
path = os.getcwd() + "/A2/graphics/"
def Metro_Hast_1D(sampleProb, deltaX, N , x0):
    x_random = np.zeros(N)
    x_random[0]  = x0
    accept = 0
    accept_arr = []
    for i in range(1,N):
        x_next = x_random[i-1] + (np.random.random() -0.5)*deltaX
        if sampleProb(x_next)/sampleProb(x_random[i-1]) > np.random.random():
            x_random[i] = x_next
            accept += 1
            
        else:
            x_random[i] = x_random[i-1]
        accept_arr.append(accept/i)
    return x_random , accept_arr
def autocorr_recursive(x):
    x = np.array(x)
    x_mean = np.mean(x)
    N = len(x)
    t = np.arange(0,N)
    S = np.zeros(N)
    S2 = np.zeros(N)
    C = np.zeros(N)
    C2 = np.zeros(N)
    S[0] = np.sum(x)
    S2[0] = np.sum(x)
    C[0] = np.sum(x*x)
    C2[0] = np.sum(x*x)
    B = (N-t)*x_mean**2
    for ti in range(1,N):
        S[ti] = S[ti-1] - x[N-ti-1]
        S2[ti] = S2[ti-1] - x[N-1]
        C[ti] = C[ti-1] - x[N-ti-1]*x[N-1]
        C2[ti] = C2[ti-1] - x[N-ti-1]**2
    result = (C - x_mean*(S + S2) + B)/(C2 - 2*x_mean*S + B)
    return result
    


def two_gaussian(x,x0):
    A = np.exp(-((x+x0)**2)/2)
    B = np.exp(-((x-x0)**2)/2)
    return (1/np.sqrt(8*np.pi))*(A+B)
def get_bayesian_expect(pi,N,nb):
    return (pi*N + 1)/(N + nb + 1)
def get_bayesian_error(expect,N,nb):
    return np.sqrt((expect*(1-expect))/(N+nb+2))
def get_bin_number(randoms,bin_width):
    N = int(np.abs(max(randoms)- min(randoms))/bin_width)
    return N
def get_bin_center(randoms, bin_number):
    array_center = np.linspace(min(randoms), max(randoms), bin_number)
    return array_center
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def main(abc = False, d = False):
    if abc:
        x_t = Metro_Hast_1D(lambda x: two_gaussian(x,2),2,100,0)
        
        
        
        
        Xi = [0,2,6]
        #plt.style.use("dark_background")
        fig_a1 , ax_a1 = plt.subplots(3,2, figsize=[22,14], sharey= False,gridspec_kw={'width_ratios': [2, 1]})
        fig_a1.tight_layout()
        #fig_a1.set_facecolor('darkgray')
        for k,j in tqdm(enumerate(Xi),desc="Xi loop",total=len(Xi)):
            xk = np.arange(-(j +4),j +4,1e-2)
            tg = two_gaussian(xk,j)
            all_paths = []
            cmap = get_cmap(50,'magma')
            #ax_a1[k,0].set_facecolor('darkgray')
            ax_a1[k,0].grid()
            for i in range(0,50):
                start = np.random.randint(-(j +4)*10,(j +4)*10)/10
                x_t,a = Metro_Hast_1D(lambda x: two_gaussian(x,j),6,400,start) 
                N = np.arange(0,len(x_t),1)
                all_paths.append(x_t)
                #ax_a1[k,0].plot(N,a)
                
                ax_a1[k,0].plot(N,x_t, marker = ".", markersize = 2, color = cmap(i))
                
            ax_a1[k,0].set_ylabel("x(t)")
            ax_a1[k,0].set_xlabel("N / iterations")
            all_paths_combined = np.concatenate(all_paths, axis=None)
            hist_all , bins_all= np.histogram(all_paths_combined, bins=20,density= True)
            
            centers = get_bin_center(all_paths_combined,20)
            exp = []
            error = []
            leng = len(all_paths_combined)

            for i in hist_all:
                nb = len(hist_all)
                exp.append(get_bayesian_expect(i,leng,nb))
                error.append(get_bayesian_error(get_bayesian_expect(i,leng,nb),leng,nb))
            #ax_a1[k,1].set_facecolor('darkgray')
            mean_all_paths = np.mean(all_paths_combined)
            std_all_paths = np.std(all_paths_combined)
            pos_std = [mean_all_paths-std_all_paths,mean_all_paths + std_all_paths]
            ax_a1[k,1].set_xlim(0,max(hist_all)*1.6)
            ax_a1[k,1].hlines(pos_std,0,max(hist_all)*1.2, "g", label = f"$\sigma$ = {std_all_paths:.2f}")
            ax_a1[k,1].hlines([mean_all_paths],0,max(hist_all)*1.2, "r", label = f"$\mu$ = {mean_all_paths:.2f}")
            ax_a1[k,1].plot(tg,xk, label = "Sample distribution")
            ax_a1[k,1].stairs(hist_all,bins_all, orientation = "horizontal", fill =True, color = cmap(45) , label = "Histogram all chains combined \n $N_i$ = 400")
            ax_a1[k,1].stairs(hist_all,bins_all, orientation = "horizontal",  color = cmap(5) )
            #ax_a1[k,1].hist(np.concatenate(all_paths, axis=None),bins = 20,density = True,orientation ="horizontal")
           
            ax_a1[k,1].set_xlabel("P / 1")
            ax_a1[k,1].set_ylabel("x(t)")
            ax_a1[k,1].errorbar(exp,centers,[0]*len(centers),error,fmt= ' ', capsize=1, color = "r", label = "bayesian error")
            ax_a1[k,1].legend()
        fig_a1.savefig(path +f"markovchain_xi_1abc_final.pdf")
            #fig_a1.clf()
    if d:
        Xi = [0,2,6]
        fig_d , ax_d = plt.subplots(3,3, figsize=[22,14], sharey= False,)
        fig_d.tight_layout()
        cmap = get_cmap(50,"ocean")
        for i,xi in tqdm(enumerate(Xi),desc="Xi loop",total=len(Xi)):
            X_markov,a = Metro_Hast_1D(lambda x: two_gaussian(x,xi),6,10000,np.random.random())
            X_markov = X_markov[:100]
            rho = sm.tsa.acf(X_markov, nlags=len(X_markov))
            N = np.arange(0,len(X_markov),1)
            autocorr_func = autocorr_recursive(X_markov)
            ax_d[i,0].plot(N,X_markov, marker = ".", markersize = 2, color = cmap(5*i))
            ax_d[i,1].hlines([0],0,100,"black")
            ax_d[i,1].plot(N,rho,color = cmap(5*i))
            #ax_d[i,1].plot(N,autocorr_func,color = cmap(10*i))
            ax_d[i,2].scatter(X_markov[:(len(N)-1)],X_markov[1:],color = cmap(5*i))
            

        fig_d.savefig(path +"autocorrelation.pdf")

    return




main(abc= False,d = True)