import numpy as np 
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import norm
import random as rd

import os
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