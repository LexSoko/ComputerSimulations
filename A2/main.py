import numpy as np 
import math as m
import matplotlib.pyplot as plt
from tqdm import tqdm
#from itertools import batched


import os

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

def binning_anal(x, kmax = None):
    if kmax == None:
        k = [2**n for n in range(round(np.log(len(x))/(np.log(2)) -1 ))]
    else:
        k = [n for n in range(1,kmax)]
    all_mean_k = np.zeros(len(k),dtype=type(np.array([])))
    for j, kj in tqdm(enumerate(k),desc="mean loop binning",total=len(k)):
        chunk_amount = (len(x)//kj)
        chunk_size = chunk_amount*kj
        blocks_k = np.array(np.split(x[:chunk_size],chunk_amount))
        means_kj = []
        for Bi in blocks_k:
            means_kj.append(np.mean(Bi))
        all_mean_k[j] = means_kj
    all_variances_k = np.zeros(len(k))
    for j, kj in tqdm(enumerate(k),desc="var loop binning",total=len(k)):
        chunk_amount = (len(x)//kj)
        all_variances_k[j] = (np.var(all_mean_k[j]))/len(all_mean_k[j])


    return all_variances_k , k    

def auto_corr_recursive(x):
    x_0 = x - np.mean(x)
    cov = np.zeros(len(x))
    cov[0] = x_0.dot(x_0)
    for i in range(len(x)-1):
            cov[i + 1] = x_0[i + 1 :].dot(x_0[: -(i + 1)])
    return cov/cov[0] 
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

def main(abc = False, d = False, e = False):
    if abc:
        x_t = Metro_Hast_1D(lambda x: two_gaussian(x,2),2,100,0) 
        Xi = [0,2,6]
        
        fig_a1 , ax_a1 = plt.subplots(3,2, figsize=[22,14], sharey= False,gridspec_kw={'width_ratios': [2, 1]})
        fig_a1.tight_layout()
        
        for k,j in tqdm(enumerate(Xi),desc="Xi loop",total=len(Xi)):
            xk = np.arange(-(j +4),j +4,1e-2)
            tg = two_gaussian(xk,j)
            all_paths = []
            cmap = get_cmap(50,'magma')
            
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
            
            mean_all_paths = np.mean(all_paths_combined)
            std_all_paths = np.std(all_paths_combined)
            pos_std = [mean_all_paths-std_all_paths,mean_all_paths + std_all_paths]

            ax_a1[k,1].set_xlim(0,max(hist_all)*1.6)
            ax_a1[k,1].hlines(pos_std,0,max(hist_all)*1.2, "g", label = f"$\sigma$ = {std_all_paths:.2f}")
            ax_a1[k,1].hlines([mean_all_paths],0,max(hist_all)*1.2, "r", label = f"$\mu$ = {mean_all_paths:.2f}")
            ax_a1[k,1].plot(tg,xk, label = "Sample distribution")
            ax_a1[k,1].stairs(hist_all,bins_all, orientation = "horizontal", fill =True, color = cmap(45) , label = "Histogram all chains combined \n $N_i$ = 400")
            ax_a1[k,1].stairs(hist_all,bins_all, orientation = "horizontal",  color = cmap(5) )
            ax_a1[k,1].set_xlabel("P / 1")
            ax_a1[k,1].set_ylabel("x(t)")
            ax_a1[k,1].errorbar(exp,centers,[0]*len(centers),error,fmt= ' ', capsize=1, color = "r", label = "bayesian error")
            ax_a1[k,1].legend()
        fig_a1.savefig(path +f"markovchain_xi_1abc_final.pdf")
            
    if d:
        Xi = [0,2,6]
        fig_d , ax_d = plt.subplots(3,3, figsize=[22,14], sharey= False,)
        fig_d.tight_layout()
        cmap = get_cmap(50,"ocean")
        for i,xi in tqdm(enumerate(Xi),desc="Xi loop",total=len(Xi)):
            X_markov,a = Metro_Hast_1D(lambda x: two_gaussian(x,xi),6,1000,np.random.random())
            X_markov = X_markov[:200]
           
            N = np.arange(0,len(X_markov),1)

            autocorr_func = auto_corr_recursive(X_markov)

            ax_d[i,0].plot(N,X_markov, marker = ".", markersize = 2, color = cmap(5*i), label = f"Markov Chain $\\xi$ = {xi}")
            ax_d[i,0].legend()
            ax_d[i,1].hlines([0],0,200,"black")
            ax_d[i,1].plot(N,autocorr_func,color = cmap(10*i), label = "$\\rho (t)$")
            ax_d[i,1].legend()
            ax_d[i,2].scatter(X_markov[:(len(N)-1)],X_markov[1:],color = cmap(5*i))
            ax_d[i,2].set_xlabel("$X_{i}$")
            ax_d[i,2].set_ylabel("$X_{i+1}$")
            

        fig_d.savefig(path +"autocorrelation_d.pdf")
 
    if e:
        
        Xi = [0,2,6]
        fig_e , ax_e = plt.subplots(3,3, figsize=[22,14], sharey= False,)
        fig_e.tight_layout()
        for i,xi in tqdm(enumerate(Xi),desc="Xi loop",total=len(Xi)):
            deltax = 6
            cmap = get_cmap(50,"winter")
            if xi == 6:
                deltax = 25
            X_markov,a = Metro_Hast_1D(lambda x: two_gaussian(x,xi),deltax,1000000,0)
            blocks_variance, k = binning_anal(X_markov,300)
            ax_e[i,0].plot(k, blocks_variance,color= cmap((i+1)*10) ,label = f"$\\xi$ = {xi}, N = {len(X_markov)}\n $\\tau$ =  {(0.5*(blocks_variance[-1]/blocks_variance[0])):.2f}")
            ax_e[i,0].legend()
            ax_e[i,1].plot(np.arange(0,len(X_markov),1),X_markov, color= cmap((i+1)*10))
            ax_e[i,2].hist(X_markov,bins=500, orientation="horizontal",color = cmap((i+1)*10) )
        fig_e.savefig(path + "binning_analysis_e1M_dx25.pdf")
                

            

    return




main(abc= False,d = False, e = True)