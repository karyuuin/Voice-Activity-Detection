import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from functionVAD import FunctionVAD

sample_rate, voice = read('usr0001_male_youth_004.wav')
samples = np.array(voice,dtype=float)
arr, arr2, arr3 = FunctionVAD(samples, sample_rate)
plt.figure(2)
plt.plot(arr)
plt.show
plt.figure(3)
plt.plot(arr2)
plt.show
plt.figure(4)
plt.plot(arr3)
plt.show
