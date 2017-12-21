import numpy as np
from sklearn.neural_network import MLPClassifier


def function2(signal,  sample_rate):
    win=[]  
    sig1=signal
    sig = np.fft.rfft(sig1)
    sig = np.abs(sig)
    l=len(sig)
    size=int(sample_rate*0.03)
    win=(np.zeros(size))/size
    for i in range(0,l,4):
        win[i:i+2]=1    
    line1=np.convolve(sig, win, mode='same')

    win=(np.zeros(size))/size
    for i in range(0,l,6):
        win[i:i+3]=1
    line2=np.convolve(sig, win, mode='same')
   
    win=(np.zeros(size))/size
    for i in range(0,l,8):
        win[i:i+4]=1
    line3=np.convolve(sig, win, mode='same')
     
    win=(np.zeros(size))/size
    for i in range(0,l,12):
        win[i:i+6]=1  
    line4=np.convolve(sig, win, mode='same')
    m1=max(line1)
    m2=max(line2)
    m3=max(line3)
    m4=max(line4)
    m=max(m1,m2,m3,m4)   
    if m>1300:  
        return True   
    else:
        return False
           
def function1(signal,   sample_rate):   
    sig=abs(signal)   
    size=int(sample_rate*0.03)
    one=(np.ones(size))/size
    line=np.convolve(sig, one,  mode='same')   
    filt=signal[np.where(line>0.5)[0]] 
    return filt


def Block(data):
    
    data = (data - np.mean(data)) / np.std(data)

    fft_size = 1024 
    X = np.array([]) 

    for i in range(0, len(data) - fft_size, fft_size): 
        block = data[i:i+fft_size] 
        block = block * np.hamming(fft_size) 
        sp = np.abs(np.fft.rfft(block, n=fft_size))[:fft_size // 2] 
        sp = sp / np.max(sp) 
        X = np.concatenate((X, sp)) 
    X = X.reshape((-1, fft_size // 2))
    return X

def Neuro(sample_rate, samples):
    
    X=Block(samples)
        
    y = np.zeros(len(X))
    
    y[30:45] = 1
    y[52:65] = 1
    y[80:84] = 1 
    y[91:122] = 1
         
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1, activation='logistic', max_iter=10000) 
    clf.fit(X, y)
    
    clf.score(X, y)
        
    X1=Block(samples)
    
    pred = clf.predict(X1)
    
    idx = np.where(pred > 0)[0] * 1024

    arr3 = np.array([samples[i:i+1024] for i in idx]).ravel()    
    
    return arr3
