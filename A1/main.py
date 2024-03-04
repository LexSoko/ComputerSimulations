import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    hist , bins = np.histogram(x_random_num_arr,bins=bins, density= True)

    return hist , bins ,x_random_num_arr

def standart_deviations_height(hist_arr, plot=False):
    std_bars= []
    for h in hist_arr:
        std_bars.append(np.std(h))
    if plot:
        plt.plot(len(hist_arr),std_bars, label = "standart deviations sample size ")
        plt.xlabel("N")
        plt.legend()
        plt.savefig("./A1/graphics/std_of_bar_heights.pdf")
        plt.cla()
    return std_bars 


def animate_hist(data,N):
    fig, ax = plt.subplots()
    def update(frame):
        ax.clear()  # Clear previous frame 
        ax.hist(data[:frame], bins=41, density= True)
        ax.plot(np.arange(0,42), [1/42]*42)  # Update histogram with new data
        ax.set_title('Sample size {}'.format(N[frame])) 
    ani = FuncAnimation(fig, update, frames=len(data), blit=False, interval=3)
    plt.show()
    return

def hist_height_distribution(L, N , plot = False):
    
    height_6_bar = []
    for i in tqdm(range(L), desc = "b1 loop"):
       hist_b1 ,_, x_random_b1 = generate_hist(N, plot= False, zero_one=True, bins = 10,  )
       #doesnt normalize the right way so a normalization factor is calculated
       w = np.sum(hist_b1)
       height_6_bar.append(hist_b1[5]/w)
    hist_height , bins = np.histogram(height_6_bar)
    w_height = np.sum(hist_height)
    if plot:
        plt.stairs(hist_height/w_height , bins,fill= True)
        plt.show()
    return hist_height, bins

def main(a1=True, b1 = True):
   
    if a1 :
        N = np.arange(10,1000,1)
        hist_arr = []
        for n in tqdm(N,desc = "a1 loop"):
            #should converge to 1/x_{i,max} for large N
            hist ,_, x_random = generate_hist(n,plot=False)
            hist_arr.append(hist)
        if True:
            animate_hist(x_random,N)    
        std_heights = standart_deviations_height(hist_arr,plot = True)
    
    if b1:
        hb1 , binb1 = hist_height_distribution(1000,1000,plot= True)
        N = [10**m for m in range(2,6)]
        hist_heights_N = []
        for n in tqdm(N):
            h , b = hist_height_distribution(1000,n)
            hist_heights_N.append(h)
        std_heights_N = standart_deviations_height
           
           
           


    return


main(a1 = False, b1 = True)