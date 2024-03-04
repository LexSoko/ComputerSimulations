import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
#b
def S_recursive(t_max,X_given):
    N_samples = len(X_given) - 1
    X = X_given
    S = np.zeros(t_max)
    S[0] = np.sum(X)
    for t in tqdm(range(1,len(S)), desc = "t loop"):
        S[t] = S[t-1] - X[N_samples-t]
    return S


def main(c = False , d = False, e = False):
    if c: 
        X = [1,2,3]
        S, X1 = S_recursive(2, X)
        print(S)
    if d:
        N = np.arange(1e7,1e8,1e7)
        print(len(N))
        timer = []
        
        for n in tqdm(N,desc= "main loop d"):
            X_d = np.random.randint(0,42,int(n))
            t_max_d = len(X_d)-2
            start = time.time()
            S_d = S_recursive(t_max_d,X_d) 
            end = time.time()
            timer.append(end - start)
        
        plt.plot(N,timer)
        plt.savefig("./A0/graphics/d_plot.pdf")
    if e:
        X_e = np.random.randint(0,42,1000)
        t_max = len(X_e)-2 
        S = S_recursive(t_max,X_e)
        S_not = []
        for t in range(0,t_max):
            S_t = 0
            for i in range(0,len(X_e)-1-t):
                S_t += X_e[i]
            S_not.append(S_t)
        plt.plot(S, label = "S recursive")
        plt.plot(S_not, "r--" ,label = "S not recursive")
        plt.savefig("./A0/graphics/just_checking.pdf")
main(d = False, e = True)