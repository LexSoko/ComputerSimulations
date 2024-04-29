import numpy as np 
import math as m
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import scipy.optimize as spo
from statsmodels.tsa.stattools import acf
import os
import scipy.signal as signal
from time import time

__author__ = "Aleksey Sokolov"
#12004091
path = os.getcwd() + "/A2/graphics/" #just delete the string
#parameters for dx optimisation/guess
std_params_lfilter = {
    "order": 3,
    "crit_f": 0.4,
    "type": 'low',
}
std_params_peaks = {
    "height": None,
    "width": 16,
    "prominence": 8,
    "threshold": None,
    "distance": None,
    "rel_height": 0.5

}
#Metro hasting algorithm for onedimensional markov chain
def Metro_Hast_1D(sampleProb, deltaX, N , x0 ,disable_pgb= True,optimize = False ):
    #if optimize is true, the function will calculate a guess for deltax and use it for the markovchain 
    if optimize:
        all_deltaX, min_index , _ = optimize_dX(sampleProb,findpeak_params = std_params_peaks,filter_params = std_params_lfilter)
        deltaX = all_deltaX[min_index][-1]  #because of the bimodal nature of the distribution only the most right peak is counted
    x_random = np.zeros(N) #initialize
    x_random[0]  = x0
    accept = 0
    accept_arr = [] #for calculating the acceptance rate for each step
    for i in tqdm(range(1,N), desc = "Metro Hasting Main Loop", disable= disable_pgb):
        x_next = x_random[i-1] + (np.random.random() -0.5)*deltaX
        #i assign the acceptance probability randomly if p_i/p_j < 1, Z_j -> Z_i 
        if sampleProb(x_next)/sampleProb(x_random[i-1]) > np.random.random():
            x_random[i] = x_next
            accept += 1
            
        else:
            x_random[i] = x_random[i-1]
        accept_arr.append(accept/i)
    return x_random , accept_arr, deltaX

def optimize_dX(sampleProb,dxrange = [1,100], Navg = 2, N_of_samples = 1000, findpeak_params = std_params_peaks, filter_params = std_params_lfilter):
    deltax = np.arange(dxrange[0],dxrange[1],1) #generating the needed dx window 
    #for reducing the noic in the autocorrelation function
    b, a = signal.butter(std_params_lfilter["order"],std_params_lfilter["crit_f"], btype= std_params_lfilter["type"])
    integrated_all = []
    #essential what i am doing here is the following:
    #i calculte the autocoralation for a given dx N_avg times and than average the result
    #because a negative or positive correlation is not desired i take the absolute value of the autocorrelation function
    #this absolute function is a measure for far from uncorrelated the data is
    #the area under this curve is proporsional to the correlation of the whole markov chain over the whole timescale for a given dx 
    for k in tqdm(range(Navg),desc = "optimizing delta x"):
        integrated = []
        for i in deltax:
            integrated.append(integral_autocorr(i,lambda x: Metro_Hast_1D(sampleProb,x,N_of_samples,0)))
        integrated_all.append(integrated)
    integrated_mean = np.array([np.mean(i) for i in np.array(integrated_all).T]).T #mean calculation of all integrals
    intmean_filter = signal.filtfilt(b, a, integrated_mean) #appliying a low passfilte to get rid of the noice
    #this function now can have pultiple minima (in our case 2)
    #i look for local minima and return them
    index, _ = signal.find_peaks(-intmean_filter,**std_params_peaks) 
    
    return deltax,index, intmean_filter

#this function thakes a handle of the metrofunction with the desired distrubution and calculates the area under the autocorrelation functin
def integral_autocorr(deltax,Metro):  
    x_m , a ,_ = Metro(deltax)
    autocorr_m = auto_corr_func(x_m)
    #looking at the absolute value of the function because every difference from zero is undesirable
    auto_integrated = np.sum(np.abs(autocorr_m))

    return auto_integrated


def binning_analysis(x, kmax = None):
    #2**n doesnt work that good
    if kmax == None:
        k = np.array([2**n for n in range(round(np.log(len(x))/(np.log(2)) -1 ))])
    else:
        k = np.array([n for n in range(1,kmax)])
    all_variances_k = np.zeros(len(k),dtype=type(np.array([]))) #array for all variance for each k
    chunk_amount = (len(x)//k) #looking vor how many subarray i can do with each k
    chunk_size = chunk_amount*k #this is the index for cutting the array
    for j, kj in tqdm(enumerate(k),desc="mean loop binning",total=len(k)):
        #i splitt the markovchain an calculate the mean of each bin, and than the variance of the whole sample
        all_variances_k[j] = np.var(np.mean(np.split(np.array(x[:chunk_size[j]]),chunk_amount[j]), axis =1), axis =0)/chunk_amount[j]
   
    
    return all_variances_k , k    

#autocorellation function
def auto_corr_func(x):
    x_0 = x - np.mean(x)
    cov = np.zeros(len(x))
    cov[0] = x_0.dot(x_0) #variance of the data with itself
    #equivalent to a convolution kind of
    for i in range(len(x)-1):
            cov[i + 1] = x_0[i + 1 :].dot(x_0[: -(i + 1)])
    return cov/cov[0] 

#distribution given
def two_gaussian(x,x0):
    A = np.exp(-((x+x0)**2)/2)
    B = np.exp(-((x-x0)**2)/2)
    return (1/np.sqrt(8*np.pi))*(A+B)

#some helper functions
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

def main(abc = False, d = False,d2 = False, e = False, f=False):
    Xi = [0,2,6]
    if abc:
        
        fig_a1 , ax_a1 = plt.subplots(3,2, figsize=[22,14], sharey= False,gridspec_kw={'width_ratios': [2, 1]})
        fig_a1.tight_layout()
        deltaX = [6,14,30]
        for k,xi in tqdm(enumerate(Xi),desc="Xi loop",total=len(Xi)):
            xk = np.arange(-(xi +4),xi +4,1e-2)
            tg = two_gaussian(xk,xi)
            all_paths = []
            cmap = get_cmap(50,'magma')
            deltax = deltaX[k]
            
            ax_a1[k,0].grid()
            for i in range(0,50):
                start = np.random.randint(-(xi +4)*10,(xi +4)*10)/10 #starting at random positions
                x_t,a, _ = Metro_Hast_1D(lambda x: two_gaussian(x,xi),deltax,400,start) 
                N = np.arange(0,len(x_t),1)
                all_paths.append(x_t) #i have 50 marokovchains 
        
                
                ax_a1[k,0].plot(N,x_t, marker = ".", markersize = 2, color = cmap(i))
                
            ax_a1[k,0].set_ylabel("x(t)")
            ax_a1[k,0].set_xlabel("N / iterations")
            all_paths_combined = np.concatenate(all_paths, axis=None) #add all paths to get a sharper histogram
            hist_all , bins_all= np.histogram(all_paths_combined, bins=20,density= True)
            
            centers = get_bin_center(all_paths_combined,20)
            exp = []
            error = []
            leng = len(all_paths_combined)
            #bayesian analysis
            for i in hist_all:
                nb = len(hist_all)
                exp.append(get_bayesian_expect(i,leng,nb))
                error.append(get_bayesian_error(get_bayesian_expect(i,leng,nb),leng,nb))
            #calulating all means and std
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
        fig_a1.savefig(path +f"markovchain_1abc_final.pdf")
            
    if d:
       
        fig_d , ax_d = plt.subplots(3,3, figsize=[22,14], sharey= False,)
        fig_d.tight_layout()
        cmap = get_cmap(50,"ocean")
        for i,xi in tqdm(enumerate(Xi),desc="Xi loop",total=len(Xi)):
            
            X_markov,a, deltaX = Metro_Hast_1D(lambda x: two_gaussian(x,xi),0,1000,np.random.random(), optimize=True)
           
            N = np.arange(0,len(X_markov),1)

            autocorr_func = auto_corr_func(X_markov)

            ax_d[i,0].plot(N,X_markov, marker = ".", markersize = 2, color = cmap(5*i), label = f"Markov Chain $\\xi$ = {xi} \n $\Delta x$ = {deltaX}")
            ax_d[i,0].legend()
            ax_d[i,1].hlines([0],0,1000,"black")
            ax_d[i,1].plot(N,autocorr_func,color = cmap(10*i), marker = ".", label = f"$\\rho (t)$ \n $\Delta x$ = {deltaX}")
            ax_d[i,1].legend()
            ax_d[i,2].plot(N[:-1],a,color = cmap(10*i), marker = ".", label = "Acceptance Rate")
            ax_d[i,2].legend()
            ax_d[i,2].set_xlabel("$N$")
            ax_d[i,2].set_ylabel("$\\frac{N_{accepted}}{N_i}$ / 1")
            

        fig_d.savefig(path +"autocorrelation_d_optimized.pdf")
        
    
    if d2:
        #inverse slope of the timeseries?
        X_markov,a, deltaX = Metro_Hast_1D(lambda x: two_gaussian(x,0),20,100000,0, optimize=True)
        autocorr_func = auto_corr_func(X_markov)
        dadt = np.diff(autocorr_func) #differentiation 
        fig_d2 , axd2 = plt.subplots(2,figsize=[16,9])
        inte = np.sum(autocorr_func)
        dadt = dadt[0:50]
        autocorr_func = autocorr_func[0:50]
        print(inte)
        axd2[0].plot(np.arange(0,len(dadt),1),1/dadt)
        axd2[1].plot(np.arange(0,len(autocorr_func),1),autocorr_func)
        axd2[1].hlines([0],0,50,"black")
        plt.show()
    if e:
        
        
        window = 0.1
        #binning analysis with optimized functions and unoptimized comparisson
        cmap = get_cmap(50,"winter")
        for j in range(2):
            fig_e , ax_e = plt.subplots(3,3, figsize=[16,9], sharey= False,)
            fig_e.tight_layout()
            optimize = True
            if j == 1:
                optimize = False
            dx = [15,6,25]
            for i,xi in tqdm(enumerate(Xi),desc="Xi loop",total=len(Xi)):
                
                
                X_markov,a,deltaX = Metro_Hast_1D(lambda x: two_gaussian(x,xi),dx[i],1000000,0, optimize=optimize)
                
                blocks_variance, k = binning_analysis(X_markov,150)
                autocorr_func = auto_corr_func(X_markov)
                #i calculate a integral till autocorellation is below 0.1, the data is not correlated after this point
                zeroindex = 0
                while True:
                    if np.abs(autocorr_func[zeroindex]) < window: #kind of slow but it works
                        break
                    zeroindex += 1

                a_corr_time = np.sum(autocorr_func[0:zeroindex])
                
                
                ax_e[i,0].set_xlabel("k")
                ax_e[i,0].plot(k, blocks_variance,color= cmap((i+1)*10) ,label = f"$\\xi$ = {xi},\n N = {len(X_markov)}\n $\\tau$ =  {(0.5*(blocks_variance[-1]/blocks_variance[0])):.2f} \n $\Delta x$ = {deltaX}")
                ax_e[i,0].legend()
                ax_e[i,1].set_ylim(0,1)
                ax_e[i,1].set_xlabel("ln(t)")
                ax_e[i,1].plot(np.log(np.arange(1,len(X_markov),1)),a, "r-", label= "Acceptance Rate")
                ax_e[i,1].plot(np.log(np.arange(1,len(X_markov)+1,1)),auto_corr_func(X_markov),color= cmap((i+1)*10), label = f"Autocorrelation function \n $\\tau_i$ = {a_corr_time:.2f}")
                ax_e[i,1].legend()
                ax_e[i,2].hist(X_markov,bins=100, orientation="horizontal",color = cmap((i+1)*10) )
            fig_e.savefig(path + f"binning_analysis_e_1M_optimized{optimize}.pdf")
            fig_e.clf()
        
    if f:
        Xi = [0,2,4,6]
        #some ploting params
        plt.style.use('bmh')
        cmap = get_cmap(50,"winter")
        fig_f , axf = plt.subplots(1,1, figsize=[18,8])
        axf.grid("both")
        
        
        for i,xi in enumerate(Xi):
            #calculates the optimal delta x and plots the deltaX Spectrum
            #intmean_filter are the certain integrals of the mean autocorralation functions for a given deltaX over the whole time frame 
            deltaX, index ,intmean_filter = optimize_dX(lambda x: two_gaussian(x,xi),Navg=20) 
            axf.plot(deltaX,intmean_filter, "-",color = cmap((i+1)*10), label = f"$\\xi$  = {xi}")
            axf.plot(deltaX[index],intmean_filter[index], ".", markersize=15,color = cmap((i+1)*10), label = f"minima at $\Delta x$ = {deltaX[index]}")
        axf.set_xlabel("$\Delta x$")
        axf.set_ylabel("$\sum_{t = 0}^{t_{max}} \\rho_{t,\Delta x}$")
        axf.legend()
        fig_f.savefig(path + f"optimal_dx_analysis_f2.pdf")
       
  
    return



if __name__ == '__main__':
    main(abc= True,d = True,d2=False, e = True, f =True)

