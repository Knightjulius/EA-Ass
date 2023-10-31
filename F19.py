import numpy as np
import math
import itertools

seed = 901
search_space = (0,1)
# Length of the string squared divided by 2 times the summation (from k to length – 1)  over the summation (from I to length – k) of Si times Si+k 
# Where Si = 2xi –1 
# Xi = the ith location of a bit 
# Si+k = 2Xi+k –1 
# Kmax = n-1 
# Imax = n-k 


def autocorrelation(sequence):
    n = len(sequence)
    autocorr = np.correlate(sequence, sequence, mode='full')
    return autocorr[n - 1:]  # We are interested in the second half of the autocorrelation result

def find_low_autocorrelation_sequence(length):
    best_sequence = None
    lowest_peak = float('inf')

    for binary_sequence in itertools.product([0, 1], repeat=length):
        autocorr = autocorrelation(binary_sequence)
        peak = max(autocorr)

        if peak < lowest_peak:
            lowest_peak = peak
            best_sequence = binary_sequence

    return best_sequence, lowest_peak

sequence_length = 10  # Change this to the desired length
best_sequence, lowest_peak = find_low_autocorrelation_sequence(sequence_length)

print("Best Sequence:", best_sequence)
print("Lowest Peak Autocorrelation:", lowest_peak)

