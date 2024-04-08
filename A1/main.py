import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm




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

def standart_deviations_height(hist_arr, plot=False):
    #calculates the strandart deviations for L Samples
    std_bars= []
    for h in hist_arr:
        std_bars.append(np.std(h))
    if plot:
        plt.plot(range(len(hist_arr)),std_bars, label = "standart deviations sample size ")   # this is nonesence , has to be reworked
        plt.xlabel("N")
        plt.legend()
        plt.savefig("./A1/graphics/std_of_bar_heights1.pdf")
        plt.cla()
    return std_bars 


def hist_height_distribution(L, N , bins_needed = (0,9), plot = False):
    
    height_selected_bars = []
    for i in tqdm(range(L), desc = "b1 loop"):
       hist_b1 ,_, x_random_b1 = generate_hist(N, plot= False, zero_one=True, bins = 10,  ) #making histograms of uniform random numbers
       #doesnt normalize the right way so a normalization factor is calculated
       
       height_selected_bars.append(hist_b1[bins_needed[0]:bins_needed[1]]) # extract selected bar
    hist_height , bins = np.histogram(np.concatenate(height_selected_bars,axis=None), bins =50)
    w_height = np.sum(hist_height) # doesnt normalise it right, so do it manualy
    if plot:
        plt.stairs(hist_height/w_height , bins,fill= True)
        plt.show()
    return hist_height/w_height, bins , height_selected_bars

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


#some helper functuons
def get_bin_number(randoms,bin_width):
    N = int(np.abs(max(randoms)- min(randoms))/bin_width)
    return N
def get_bin_center(randoms, bin_number):
    array_center = np.linspace(min(randoms), max(randoms), bin_number)
    return array_center
#baysian expectation value an error
def get_bayesian_expect(Ni,N,nb):
    return (Ni + 1)/(N + nb + 1)
def get_bayesian_error(expect,N,nb):
    return np.sqrt((expect*(1-expect))/(N+nb+2))


#frequentist errors
def bernoulli_error(Ni, N):
    return np.sqrt(Ni*(1- Ni/N)/N)/np.sqrt(N)
def bernoulli_error2(pi,N):
    return np.sqrt(pi*(1-pi))/np.sqrt(N)
#cauchy distribution problem 2
def cauchy(x):
    return (1/np.pi)*(1/(1+x**2))
#probability dist from problem 3
def prob_dist(x):
    return (1/np.pi)*(2/(1-np.exp(-2)))*(np.sin(x)**2)/(1+x**2)
#inverse cdf of the cauchy distribution
def inverse_cdf(gamma):
    return np.tan((gamma-0.5)*np.pi)

exp_dist_right = lambda x ,lam: np.where( x< 0.5 ,0.5*lam*np.exp(lam*x),  lam * 0.5*np.exp(-lam*x)) #exp(-|x|) distribution from problem 3
inverse_exp_right = lambda x, lam: np.where(x<0.5, np.log(2*x)/lam, -np.log(2*(1-x))/lam) #inverse cdf of the exp distribution

def main(a1=False, b1 = False, c1 = False , a2 = False, a3= False, b3=False,a4 = False):
   
    if a1 :
        N = np.arange(100,10000,1)
        hist_arr = []
        for n in tqdm(N,desc = "a1 loop"):
            #should converge to 1/x_{i,max} for large N
            hist ,_, x_random = generate_hist(n,plot=False, zero_one= True)
            hist_arr.append(hist)

        std_heights = standart_deviations_height(hist_arr,plot = False)
        plt.plot(N,std_heights, label = "standart deviations sample size ")
        #plt.plot(N, 3/(10*np.sqrt(N)), label = "$\\frac{3}{10 \sqrt{N}}$")  # factor 1/5 for standart deviation  TODO= look into this
        plt.xlabel("N")
        plt.legend()
        plt.savefig("./A1/graphics/std_of_bar_heights_1a.pdf")
        plt.cla()
    
    if b1: 
        N = [10**m for m in range(2,7)] #generate samples with given sizes sizes 
       
        fig_gauss, axis = plt.subplots(5,1, figsize=[18, 22])
        x = np.arange(0,0.3,1e-3)
        hist_heights_N = []
        
        for i, n in tqdm(enumerate(N)):
            
            h , b, height_random = hist_height_distribution(1000,n,bins_needed= (5,6)) #different sample sizes N , each height from 1000 samples,sixth bin
            axis[i].set_xlim(0.025, 0.175)
            axis[i].stairs(h , b,fill= True, label= f"Sample Size N = {n} \n Samples L = 1000")
            axis[i].legend()
            hist_heights_N.append(height_random) #sixth bin height destribution
           

        fig_gauss.savefig("./A1/graphics/all_heights_sixth_bin_1b.pdf")
        fig_gauss.clf()
        std_heights = standart_deviations_height(hist_heights_N) # standart deviations of the heights of the sixth bin
        plt.plot
        plt.plot(np.log10(N),std_heights, label="standard deviation of the height of the sixth bar ")
        plt.xlabel("$\log_{10}(N)$ ")
        plt.legend()
        plt.savefig("./A1/graphics/std_of_sixth_bar_heights_1b.pdf")
            
    if c1:
        N_c1 = [69,420,1000,6969] #different sample sizes
        
        hists_c1 = [generate_hist(n,range_randint=[0,50], bins=5)[0:2] for n in N_c1] #for each there is a uniform disribution generated
    
        errors = [] #erroarray for frequentist analysis
        errors_pos = [] #to plot the errors at the positions of alle histograms samples
        for j,hist_c1 in enumerate(hists_c1):
            error = []
            err_pos = []
            for i ,pi in enumerate(hist_c1[0]):
                error.append(bernoulli_error2(pi,N_c1[j]))#frequentist analysis
                err_pos.append(hist_c1[1][i+1]- 5) #saving positions for errorbars
            
            errors_pos.append(err_pos) #added to the total array
            errors.append(error)

        fig_c1, ax_c1 = plt.subplots(2,2, figsize=[18,18])
        #plot
        for i ,(error,err_pos, hist_c1) in enumerate(zip(errors,errors_pos,hists_c1)):
            ax_c1[i//2,i%2].stairs(hist_c1[0],hist_c1[1],fill=True,label = f"N = {N_c1[i]}")
            ax_c1[i//2,i%2].errorbar(err_pos,hist_c1[0],error,[0]*len(hist_c1[0]), "r.", label=f"estimated error \n error_avg = {np.mean(error):2f}") 
            ax_c1[i//2,i%2].legend()
            
        fig_c1.savefig("./A1/graphics/plot_estimated_error_1c.pdf")
        fig_c1.clf()
    
    if a2:
        sample_count = 40
        bin_size = 0.5
        bin_number = get_bin_number(cauchy_total, bin_size)
        
        
        uniform_random = [generate_hist(100,zero_one=True)[2] for i in range(sample_count)] #uniform random numbers
        cauchy_sample = [inverse_cdf(u) for u in uniform_random] #the inverse of the cumulutative function for each sample
        sample_arr = [[s]*len(uniform_random[s]) for s in range(sample_count)] #so that each sample can be identified
        
        cauchy_total = np.concatenate(cauchy_sample,axis=None) #merge all samples
        x = np.arange(-50,50,1e-2)
        
        #plot, first the uniform random samples, middle inverse of that samples, right the resulting distribution each sample summed 
        fig_a2, ax_a2 = plt.subplots(1,3, figsize=[18,18])
        ax_a2[0].scatter(sample_arr,np.concatenate(uniform_random,axis=None),s = 0.1)
        ax_a2[0].set_xlabel("Samples / L")
        ax_a2[0].set_ylabel("Random Variable / $X$")
        ax_a2[1].set_ylim(-30,30) 
        ax_a2[1].scatter(sample_arr,np.concatenate(cauchy_sample,axis=None) ,s = 0.1) 
        ax_a2[1].set_ylabel("$G^{-1}$" )
        ax_a2[1].set_xlabel("Samples / L")
        ax_a2[2].set_ylim(-30,30)         
        ax_a2[2].hist(np.concatenate(cauchy_sample,axis=None), density=True, bins = bin_number,orientation ="horizontal")
        ax_a2[2].plot(cauchy(x),x, label ="$\\frac{1}{\pi} \\frac{1}{1+x^2}$")  
        ax_a2[2].legend()
        fig_a2.savefig("./A1/graphics/cauchy_dist_mulitple_samples_2a.pdf")
            
    if a3:
        #using rejection method with cauchy
        random_a3 = reject_method(lambda x: prob_dist(x), 10000,
                                  env0 = lambda x: cauchy(x),
                                  trans0 = lambda x: inverse_cdf(x),
                                  int0 = [-30,30],
                                    c0 = 2/(1-np.exp(-2))*1.02, 
                                  pdf_name="rejection_method_a3",
                                  bin_size= 0.5,
                                  save=True,
                                  )
    if b3:
        #kind of scetchy beause exp(-|x|) is discontinuous at x = 0, but hey it works
        exp_envelope = reject_method(lambda x: prob_dist(x),10000, pdf_name= "rejection_method_exp_3b", bin_size=0.5, save=True,
                                env0= lambda x: exp_dist_right(x,0.5),
                                trans0 = lambda x: inverse_exp_right(x,0.5),
                                inter0 = [-50,50],
                                c0 = 3,)

    if a4:
        N= [100,1000,10000,100000]
        #calculating random samples from the methods
        for n in N:
            cauchy_inverse = inverse_method(lambda x: inverse_cdf(x), n)

            cauchy_envelope = reject_method( lambda x: prob_dist(x), n, pdf_name= "test8",bin_size=1,
                                env0= lambda x: cauchy(x),
                                trans0 = lambda x: inverse_cdf(x),
                                inter0 = [-50,50],
                                c0 =  2/(1-np.exp(-2))*1.02,)
            print("acceptance rate should be ", 1/(2/(1-np.exp(-2))*1.02)) #theoretical acceptance rate

            exp_envelope = reject_method(lambda x: prob_dist(x),n, pdf_name= "test_cond", bin_size=0.5, save=True,
                                env0= lambda x: exp_dist_right(x,0.5),
                                trans0 = lambda x: inverse_exp_right(x,0.5),
                                inter0 = [-50,50],
                                c0 = 3,)
            print("acceptance rate should be ", 1/3) #theoretical acceptance rate
            generators = [cauchy_inverse,cauchy_envelope,exp_envelope] 
            generators_names = ["cauchy inverse method","cauchy reject method","exp reject method"]
            distributions = [lambda x: cauchy(x), lambda x: prob_dist(x),lambda x: prob_dist(x)]
            histograms = []
            bin_width = 1
            
            fig_a4, ax_a4 = plt.subplots(3,2, figsize=[18,18])
            #plotting all
            for i, (g2,d) in tqdm(enumerate(zip(generators,distributions)),desc = "generators loop"):
                x = np.arange(-20,20,1e-2)
                g = [i for i in g2 if i < 20 and i > -20]
                N = 40
                
                bin_width = np.abs(max(g)-min(g))/N 
                print("binswidth", bin_width)
                print("len after filtering ", len(g), "and before ", len(g2))
                center = get_bin_center(g,N)
                hist , bin = np.histogram(g, bins=N)
                
               
                hist2 , bin2 = np.histogram(g, bins=N, density= True)
                frequentist_err = []
                baysian_errors = []
                bayesian_expect = []
                #calculaltiing the errors , with each method
                for h in tqdm(hist,desc="hist loop"):
                    frequentist_err.append(bernoulli_error(h,len(g)))
                    bay_exp = get_bayesian_expect(h,len(g),N)
                    bayesian_expect.append(bay_exp)
                    baysian_errors.append(get_bayesian_error(bay_exp, len(g), N))
                #comparing frequentist (left) and bayesian (right)
                ax_a4[i,0].stairs(hist2,bin2,label=f"frequentist {generators_names[i]} \n N = {len(g)} ", fill=True)
                ax_a4[i,0].plot(x,d(x))
                ax_a4[i,0].errorbar(center,hist2,frequentist_err, [0]*len(hist), fmt= ' ', capsize=1)
                ax_a4[i,0].set_xlim(-20,20)
                ax_a4[i,0].legend()
                ax_a4[i,1].stairs(hist2,bin2,label=f"bayesian {generators_names[i]} \n N = {len(g)}",fill=True)
                ax_a4[i,1].plot(x,d(x))
                ax_a4[i,1].errorbar(center, np.array(bayesian_expect)/bin_width, np.array(baysian_errors)/bin_width, [0]*len(hist),fmt= ' ', capsize=1)
                ax_a4[i,1].set_xlim(-20,20)
                ax_a4[i,1].legend()
            fig_a4.suptitle(f'N = {n}')
            fig_a4.savefig(f"./A1/graphics/bayesian_frequent_N{n}_4a.pdf")
            fig_a4.clf()

    return

main(a1 = False, b1 = False, c1 = False, a2 = False, a3 = False, b3=False, a4 = False)



