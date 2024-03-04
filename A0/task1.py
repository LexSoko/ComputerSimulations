import numpy as np
import matplotlib.pyplot as plt


sensitivity_1 = 0.938
pcr_positive = 35


#I & II
false_negatives = round(pcr_positive * (1 - sensitivity_1))
print(f"Number of False Negatives: {false_negatives:.0f} +- {np.sqrt(false_negatives):.0f}")

#III
def number_for_prop_bin(std,prob):
    return round(1/(std**2/(prob*(1-prob))))
sensitivity_2 = 0.94

number_for_01 = number_for_prop_bin(0.001,0.94)
number_for_1 = number_for_prop_bin(0.01,0.94)
print(f"tested for std 0.1 : {number_for_01} ")
print(f"tested for std 1 : {number_for_1} ")


