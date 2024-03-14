import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter, FFMpegWriter
from tqdm import tqdm
from scipy.stats import norm
import random as rd
from matplotlib import pyplot, transforms
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
    hist , bins = np.histogram(x_random_num_arr,bins=bins, density= True)
    w = np.sum(hist)
    return hist/w , bins ,x_random_num_arr

def standart_deviations_height(hist_arr, plot=False):
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


def animate_hist(data,N, range_randint = [0,42]):
    fig, ax = plt.subplots()
    N_ = np.arange(0,N,10, dtype=int)
    def update(frame):
        ax.clear()  # Clear previous frame 
        ax.hist(data[:frame],  density= True)
        #ax.plot(np.arange(range_randint[0],range_randint[1]), [1/range_randint[1]]*range_randint[1])  # Update histogram with new data
        ax.set_title('Sample size {}'.format(frame)) 
    ani = FuncAnimation(fig, update, frames=len(data), blit=False, interval=1)
    plt.show()
    pbar = tqdm(desc="mp4-saving", total=0)
    class Framesaver:
    # helper class to use last_frame in update function
        last_frame = 0
    fs = Framesaver()
    # update function for the progress bar
    def update_progress(cur_fr: int, tot_fr: int):
        if cur_fr == 0:
            pbar.reset(tot_fr)
        fr_diff = cur_fr - fs.last_frame
        fs.last_frame = cur_fr
        pbar.update(fr_diff)
    writermp4 = FFMpegWriter(fps=25)
    ani.save(
        "./A1/graphics/hist_ani_3.mp4",
        writer=writermp4,
        progress_callback=update_progress,
    )
    return

def hist_height_distribution(L, N , bins_needed = (0,9), plot = False):
    
    height_selected_bars = []
    for i in tqdm(range(L), desc = "b1 loop"):
       hist_b1 ,_, x_random_b1 = generate_hist(N, plot= False, zero_one=True, bins = 10,  )
       #doesnt normalize the right way so a normalization factor is calculated
       
       height_selected_bars.append(hist_b1[bins_needed[0]:bins_needed[1]])
    hist_height , bins = np.histogram(np.concatenate(height_selected_bars,axis=None))
    w_height = np.sum(hist_height)
    if plot:
        plt.stairs(hist_height/w_height , bins,fill= True)
        plt.show()
    return hist_height/w_height, bins , height_selected_bars

def reject_method(generator,N, **kwargs ):
    leng = len( [key for key in kwargs.keys() if key[0] == "i"])
    x_total = np.zeros(leng,dtype=type(np.array([])))
    c_array = np.zeros(leng)
    env_array = np.zeros(leng, dtype=type(lambda x: x))
    transforms_array = np.zeros(leng, dtype=type(lambda x: x))
    interval_array = []
    count_1 = 0
    for key, item in kwargs.items():
        if key[0] == 'i':
            index = int(float(key[-1]))
            x_total[index] = np.arange(item[0],item[1],1e-2)
            count_1 += 1
        if key[0] == 'c':
            c_array[int(key[-1])] = item
        if key[0] == 'e':
            env_array[int(key[-1])] = item
        if key[0] == "t":
            transforms_array[int(key[-1])] = item
    random_enveloped = []
    env_analytical = []
    for env, tr, x , c in zip(env_array,transforms_array,x_total, c_array):
        x_random_1 = generate_hist(N,zero_one=True)[2]
        u = generate_hist(N,zero_one=0)[2]
        x_t = tr(x_random_1)
        
        for xt, r in zip(x_t,u):
            if c*r*env(xt) < generator(xt):
                random_enveloped.append(xt)
        env_analytical.append(env(x))
    env_analytical = np.concatenate(env_analytical,axis=None)    
    x_total = np.concatenate(x_total,axis=None)
    fig_b3, ax_b3 = plt.subplots(1,1,figsize=[18,18])
    ax_b3.hist(random_enveloped,bins=100, density=True)
    ax_b3.plot(x_total,env_analytical)
    ax_b3.plot(x_total,generator(x_total))
    #ax_b3.hist(x_1,bins=100, density=True,histtype="step")
    fig_b3.savefig("./A1/graphics/rejection_method_exp_dist_b3_new.pdf")
    return random_enveloped
        

def gaussian(x,mu,std):
    return np.exp((-1/2)*((x-mu)/std)**2)/np.sqrt(2*np.pi*std**2)

def bernoulli_error(Ni, N):
    return np.sqrt(Ni*(1- Ni/N))/N
def bernoulli_error2(pi,N):
    return np.sqrt(pi*(1-pi))/np.sqrt(N)
def cauchy(x):
    return (1/np.pi)*(1/(1+x**2))
def prob_dist(x):
    return (1/np.pi)*(2/(1-np.exp(-2)))*(np.sin(x)**2)/(1+x**2)
def inverse_cdf(gamma):
    return np.tan((gamma-0.5)*np.pi)
def exp_dist(x,lam):
    return 2*(np.exp(lam*x))
def inverse_exp(x,lam):
    return (1/lam) * np.log(1-x)
def main(a1=True,ani_a1 = False, b1 = False, c1 = False , a2 = False, a3= False, b3=False,a4 = True):
   
    if a1 :
        N = np.arange(100,10000,1)
        hist_arr = []
        for n in tqdm(N,desc = "a1 loop"):
            #should converge to 1/x_{i,max} for large N
            hist ,_, x_random = generate_hist(n,plot=False)
            hist_arr.append(hist)
        if ani_a1 == True:
            animate_hist(x_random,N[-1:])    
        std_heights = standart_deviations_height(hist_arr,plot = False)
        plt.plot(N,std_heights, label = "standart deviations sample size ")
        plt.plot(N, 1/(5*np.sqrt(N)), label = "$\\frac{1}{5 \sqrt{N}}$")  # factor 1/5 for standart deviation  TODO= look into this
        plt.xlabel("N")
        plt.legend()
        plt.savefig("./A1/graphics/std_of_bar_heights_vs_expected_try.pdf")
        plt.cla()

    
    if b1:
        hb1 , binb1,rand = hist_height_distribution(1000,1000,bins_needed= (5,6) ,plot= True)
        animate_hist(rand, len(rand))
        N = [10**m for m in range(2,7)]
        print(N)
        fig_gauss, axis = plt.subplots(5,1, figsize=[18, 22])
        x = np.arange(0,0.3,1e-3)
        hist_heights_N = []
        
        for i, n in tqdm(enumerate(N)):
            
            h , b, height_random = hist_height_distribution(1000,n,bins_needed= (5,6))
            gauss = gaussian(x,np.mean(height_random), np.std(height_random))
           
            axis[i].stairs(h , b,fill= True)
            axis[i].plot(x,gauss)
            hist_heights_N.append(height_random)
           

        fig_gauss.savefig("./A1/graphics/gauss.pdf")
        fig_gauss.clf()
        std_heights = standart_deviations_height(hist_heights_N)
        plt.plot(np.log10(N),std_heights, label="standard deviation of the height of the sixth bar ")
        plt.xlabel("$\log_{10}(N)$ ")
        plt.legend()
        plt.savefig("./A1/graphics/std_of_sixth_bar_heights.pdf")
            
    if c1:
        N_c1 = [69,420,1000,6969]
        
        hists_c1 = [generate_hist(n,range_randint=[0,50], bins=5)[0:2] for n in N_c1]
        print(len(generate_hist(1000,range_randint=[0,50], bins=5)[0:2][0]))
        errors = []
        errors_pos = []
        for j,hist_c1 in enumerate(hists_c1):
            error = []
            err_pos = []
            for i ,pi in enumerate(hist_c1[0]):
                error.append(bernoulli_error2(pi,N_c1[j]))
                err_pos.append(hist_c1[1][i+1]- 5)
            errors_pos.append(err_pos)
            errors.append(error)

        fig_c1, ax_c1 = plt.subplots(2,2, figsize=[18,18])
        
        for i ,(error,err_pos, hist_c1) in enumerate(zip(errors,errors_pos,hists_c1)):
            ax_c1[i//2,i%2].stairs(hist_c1[0],hist_c1[1],fill=True,label = f"N = {N_c1[i]}")
            ax_c1[i//2,i%2].errorbar(err_pos,hist_c1[0],error,[0]*len(hist_c1[0]), "r.", label=f"estimated error \n error_avg = {np.mean(error):2f}") 
            ax_c1[i//2,i%2].legend()
            
        fig_c1.savefig("./A1/graphics/plot_estimated_error_1c.pdf")
        fig_c1.clf()
    
    if a2:
        sample_count = 40
        uniform_random = [generate_hist(100,zero_one=True)[2] for i in range(sample_count)]

        cauchy_sample = [inverse_cdf(u) for u in uniform_random]
        fig_a2, ax_a2 = plt.subplots(1,3, figsize=[18,18])
        
        sample_arr = [[s]*len(uniform_random[s]) for s in range(sample_count)]
        
        x = np.arange(-50,50,1e-2)

        ax_a2[0].scatter(sample_arr,np.concatenate(uniform_random,axis=None),s = 0.1)
        ax_a2[0].set_xlabel("Samples / L")
        ax_a2[0].set_ylabel("Random Variable / $X$")
        ax_a2[1].set_ylim(-30,30) 
        ax_a2[1].scatter(sample_arr,np.concatenate(cauchy_sample,axis=None) ,s = 0.1) 
        ax_a2[1].set_ylabel("$G^{-1}$" )
        ax_a2[1].set_xlabel("Samples / L")
        ax_a2[2].set_ylim(-30,30)         
        ax_a2[2].hist(np.concatenate(cauchy_sample,axis=None), density=True, bins = 10000,orientation ="horizontal")
        ax_a2[2].plot(cauchy(x),x, label ="$\\frac{1}{\pi} \\frac{1}{1+x^2}$")  
        ax_a2[2].legend()
        fig_a2.savefig("./A1/graphics/cauchy_dist_mulitple_samples.pdf")
            
    if a3:
        x = np.arange(-50,50,1e-2)
        cauch_dist = cauchy(x)
        prob = prob_dist(x)
    
        c = 2/(1-np.exp(-2))*1.02
        _,_, u = generate_hist(10000,zero_one=True)
        x_t = inverse_cdf(generate_hist(10000,zero_one=True)[2])
        rejection_random = []
        
        for r,xt in zip(u,x_t):
            if r*c*cauchy(xt) < prob_dist(xt):
                rejection_random.append(xt)
        fig_a3, ax_a3 = plt.subplots(1,1,figsize=[18,18])
        ax_a3.set_xlim(-20,20)
        ax_a3.hist(rejection_random, bins=10000, density=True)
        
        ax_a3.plot(x,prob)
        ax_a3.plot(x,cauch_dist*c)
        #x = np.arange(0,50,1e-2)
        #plt.plot(x,exp_dist(x,-0.5))
        plt.show()
        #fig_a3.savefig("./A1/graphics/rejection_method_a3.pdf")
    if b3:
        x1 = np.arange(-50,0,1e-2)
        x2 = np.arange(0,50,1e-2)
        _,_, u1 = generate_hist(10000,zero_one=True)
        x_random_1 = generate_hist(5000,zero_one=True)[2]
        x_random_2 = generate_hist(5000,zero_one=True)[2]
        x_1 = inverse_exp(x_random_1,0.5)
        x_2 = inverse_exp(x_random_2,-0.5)
        x_t = np.concatenate([x_1,x_2],axis=None)
        c=2
        rejection_random_b3 = []
        for r , xt in zip(u1,x_t):
            if xt < 0:
                
                if -c*r*exp_dist(xt,0.5) < prob_dist(xt):
                    rejection_random_b3.append(xt)
            if xt >= 0:
                
                if c*r*exp_dist(xt,-0.5) < prob_dist(xt):
                    rejection_random_b3.append(xt)
        x_total = np.concatenate([x1,x2],axis=None)
        env_dist = np.concatenate([np.exp(x1/2)/2,np.exp(-x2/2)/2])
        fig_b3, ax_b3 = plt.subplots(1,1,figsize=[18,18])
        ax_b3.hist(rejection_random_b3,bins=100, density=True)
        ax_b3.plot(x_total,env_dist)
        ax_b3.plot(x_total,prob_dist(x_total))
        #ax_b3.hist(x_1,bins=100, density=True,histtype="step")
        fig_b3.savefig("./A1/graphics/rejection_method_exp_dist_b32.pdf")
        randi = reject_method(lambda x: prob_dist(x),5000, 
                              env0= lambda x: exp_dist(x,0.5),
                              trans0 = lambda x: inverse_exp(x,0.5),
                              inter0 = [-50,0],
                              c0 = 2,
                              env1= lambda x: exp_dist(x,-0.5),
                              trans1 = lambda x: inverse_exp(x,-0.5),
                              inter1 = [0,50],
                              c1 = 2,
                              )

    if a4:
        g=1
        def reject_metho2d(generator,N, **kwargs ):
            leng = len( [key for key in kwargs.keys() if key[0] == "i"])
            x_total = np.zeros(leng,dtype=type(np.array([])))
            c_array = np.zeros(leng)
            env_array = np.zeros(leng, dtype=type(lambda x: x))
            transforms_array = np.zeros(leng, dtype=type(lambda x: x))
            interval_array = []
            count_1 = 0
            for key, item in kwargs.items():
                if key[0] == 'i':
                    index = int(float(key[-1]))
                    x_total[index] = np.arange(item[0],item[1],1e-2)
                    count_1 += 1
                if key[0] == 'c':
                    c_array[int(key[-1])] = item
                if key[0] == 'e':
                    env_array[int(key[-1])] = item
                if key[0] == "t":
                    transforms_array[int(key[-1])] = item

            random_enveloped = []
            env_analytical = []
            for env, tr, x , c in zip(env_array,transforms_array,x_total, c_array):

                x_random = generate_hist(N,zero_one=True)[2]
                u = generate_hist(N,zero_one=0)[2]
                x_t = tr(x_random)
                
                for xt, r in zip(x_t,u):
                    if c*r*env(xt) < generator(xt):
                        random_enveloped.append(xt)
                env_analytical.append(env(x))
            env_analytical = np.concatenate(env_analytical,axis=None)    
            x_total = np.concatenate(x_total,axis=None)
            fig_b33, ax_b33 = plt.subplots(1,1,figsize=[18,18])
            ax_b33.hist(random_enveloped,bins=100, density=True)
            ax_b33.plot(x_total,env_analytical)
            ax_b33.plot(x_total,generator(x_total))
            #ax_b3.hist(x_1,bins=100, density=True,histtype="step")
            fig_b33.savefig("./A1/graphics/rejection_method_exp_dist_b3_new.pdf")
            return random_enveloped
                #def reject_method(generator,N, **kwargs ):
                #    leng = len( [key for key in kwargs.keys() if key[0] == "i"])
                #    x_total = np.zeros(len( leng),dtype=type(np.array([])))
                #    c_array = np.zeros(leng)
                #    env_array = np.zeros(leng, dtype=type(lambda x: x))
                #    transforms_array = np.zeros(leng, dtype=type(lambda x: x))
                #    interval_array = []
                #    count_1 = 0
                #    for key, item in kwargs.items():
                #        if key[0] == 'i':
                #            index = int(float(key[-1]))
                #            x_total[index] = np.arange(item[0],item[1],1e-2)
                #            count_1 += 1
                #        if key[0] == 'c':
                #            c_array[int(key[-1])] = item
                #        if key[0] == 'e':
        #            env_array[int(key[-1])] = item
        #        if key[0] == "t":
        #            transforms_array[int(key[-1])] = item
        #    for env, tr, x , c in zip(env_array,transforms_array,x_total, c_array):
        #        
        #        x_random_1 = generate_hist(N,zero_one=True)[2]
        #        u = generate_hist(N,zero_one=0)[2]
        #        x_t = tr(x_random_1)
        #        random_enveloped = []
        #        if c*u*env(x_t) < generator(x_t):
        #            random_enveloped.append(x_t)
        #        env_analytical = np.concatenate([env(x) for x in x_total])
        #        x_total = np.concatenate(x_total,axis=None)
        #        fig_b3, ax_b3 = plt.subplots(1,1,figsize=[18,18])
        #        ax_b3.hist(random_enveloped,bins=100, density=True)
        #        ax_b3.plot(x_total,env_analytical)
        #        ax_b3.plot(x_total,generator(x_total))
        #        #ax_b3.hist(x_1,bins=100, density=True,histtype="step")
        #        fig_b3.savefig("./A1/graphics/rejection_method_exp_dist_b3_new.pdf")
#
        #        
#






        #reject_method(1, i0=(2,2),e2=2,i1=(2,1))




            
            
       
       

           
           
           


    return


main(a1 = False, b1 = False, c1 = False, a2 = False, a3 = False, b3=True, a4 = True)