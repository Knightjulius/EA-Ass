import numpy as np
import math

seed = 901
search_space = (0,1)
# Length of the string squared divided by 2 times the summation (from k to length – 1)  over the summation (from I to length – k) of Si times Si+k 
# Where Si = 2xi –1 
# Xi = the ith location of a bit 
# Si+k = 2Xi+k –1 
# Kmax = n-1 
# Imax = n-k 

# bullshit
n = 100
k = n-1
i = n-k

for n in range(n):
    for k in range(n-1):
        for i in range(n-k):
            do something