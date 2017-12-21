import numpy as np
import matplotlib.pyplot as plt
from metodVAD import function2
from metodVAD import function1
from metodVAD import Neuro

def FunctionVAD(samples, sample_rate):
    signal=(samples-np.mean(samples))/np.std(samples)
    arr=[]   
    plt.figure(1)
    plt.plot(signal)
    plt.show()
    
    for i in range(0,len(signal),1024):
        flag=False
        flag=function2(signal[i:i+1024], sample_rate)
        if flag==True:
            arr[i:i+1024]=(signal[i:i+1024])
   
    arr2=function1(signal, sample_rate)
    
    arr3 = Neuro(sample_rate, samples)
    
    return arr, arr2, arr3
