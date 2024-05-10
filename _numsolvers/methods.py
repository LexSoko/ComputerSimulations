import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as signal
from tqdm import tqdm
from scipy.stats import norm
import random as rd

import os
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

def generate_hist(N,plot = False,range_randint = [0,42],bins = 41, zero_one = False):
    if zero_one:
        x_random_num_arr = np.random.random(N)
        range_randint = [0,1]
    else:
        x_random_num_arr = np.random.randint(range_randint[0],range_randint[1],N)
    if plot:
        plt.hist(x_random_num_arr,density=True,bins=bins, label= f"Sample Size: {N}")
        plt.xlabel("$x_i$")
        plt.plot(np.linspace(range_randint[0],range_randint[1],N), [1/(range_randint[1]-range_randint[0])]*len(x_random_num_arr))
        plt.legend()
        plt.show()
    hist , bins = np.histogram(x_random_num_arr,bins=bins)
    w = np.sum(hist)
    return hist/w , bins ,x_random_num_arr


def reject_method(generator,N,pdf_name = "rejection_mode",bin_size= 2, save = False, **kwargs ):
    #method that takes in a function to sample from via rejection 
    #takes also a part defined envelope function, that gets sampled with inverse method
    #key words arguments should have as first letter 
    #e..j for envelope, c..j for enveloping constants, i..j for interval for the part defined funcs, t..j for transform (inverse of the cdf) 

    leng = len([key for key in kwargs.keys() if key[0] == "i"]) #determines the amount of function parts

    #initializing the keyword arrays with the correct datatype
    x_total = np.zeros(leng,dtype=type(np.array([]))) 
    c_array = np.zeros(leng)
    env_array = np.zeros(leng, dtype=type(lambda x: x))
    transforms_array = np.zeros(leng, dtype=type(lambda x: x))
    
    for key, item in kwargs.items():
        #adding the itemes of the keywordarguments to the right arrays for further processing
        if key[0] == 'i':
            index = int(float(key[-1]))
            x_total[index] = np.arange(item[0],item[1],1e-2) 
        if key[0] == 'c':
            c_array[int(key[-1])] = item
        if key[0] == 'e':
            env_array[int(key[-1])] = item
        if key[0] == "t":
            transforms_array[int(key[-1])] = item        
    
    random_enveloped = [] #innitializing the array for the rejection method random numbers
    env_analytical = [] ##array for the function parts pdf
    
    for env, tr, x , c in tqdm(zip(env_array,transforms_array,x_total, c_array),desc="piecing envs together", total=len(c_array)):
        #two independent generated uniform random number arrays, for the rejection r*c*h(x_random)
        _,_,x_random_1 = generate_hist(N,zero_one=True)
        _,_,u = generate_hist(N,zero_one=True)
       
        x_t_inv = tr(x_random_1) #calculating the inverse of the func part cdf
        accept = 0 #counting for the acceptance rate
        for xt_inv, r in tqdm(zip(x_t_inv,u),"rejecting random variables", total=len(u)):
            #rejection method implemented
            if c*r*env(xt_inv) < generator(xt_inv):
                #when the point evaluated at the random numbers position 
                #times a random number between 0 and 1 is inside the sampled function it will be saved
                random_enveloped.append(xt_inv) 
                accept += 1
        #all analytical functions are calculated
        env_analytical.append(c*env(x))
    print("acceptance rate: ",accept/(N*len(env_array)) )
    env_analytical = np.concatenate(env_analytical,axis=None)    #merging all evaluted fucntion
    x_total = np.concatenate(x_total,axis=None) #mergind the definition space

    bin_number = int(np.abs(max(random_enveloped)- min(random_enveloped))/bin_size) # determine the bin number based on the given binsize
    #saves plot
    if save:
        fig_b, ax_b = plt.subplots(1,1,figsize=[18,18])
        ax_b.hist(random_enveloped,bins=bin_number, density=True, label =f"Initial sample size: {N*leng} \n Rejection sample size: {accept}")
        ax_b.set_xlim(min(x_total),max(x_total))
        ax_b.plot(x_total,env_analytical, label = "Enveloping function")
        ax_b.plot(x_total,generator(x_total), label = "Rejection function")
        ax_b.legend()
        fig_b.savefig(f"./A1/graphics/{pdf_name}.pdf")
    return random_enveloped

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








def inverse_method(inverse_cdf_func, N):
    func_random = inverse_cdf_func(generate_hist(N,zero_one=True)[2])       
    return func_random


def get_bayesian_expect(Ni,N,nb):
    return (Ni + 1)/(N + nb + 1)
def get_bayesian_error(expect,N,nb):
    return np.sqrt((expect*(1-expect))/(N+nb+2))
def get_bin_number(randoms,bin_width):
    N = int(np.abs(max(randoms)- min(randoms))/bin_width)
    return N
def get_bin_center(randoms, bin_number):
    array_center = np.linspace(min(randoms), max(randoms), bin_number)
    return array_center