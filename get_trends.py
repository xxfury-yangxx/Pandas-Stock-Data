import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from sklearn.cluster import KMeans
from datetime import datetime,timedelta

from scipy.stats import gaussian_kde,norm,t
from arch import arch_model

def get_trends(c,lookback):
    '''
    :type c: ndarray
    :param c: Daily Close prices

    :type lookback: int
    :param lookback: Lookback window for smoothing
    '''
    cs = pd.Series(c)
    ema = cs.ewm(span=lookback).mean()
    ema = ema[::-1].ewm(span=lookback).mean()[::-1]
    ema = ema.values
    
    lr = np.diff(np.log(ema))        
    km = KMeans(3).fit(lr.reshape(-1,1))
    
    lb = km.labels_
    
    #Change labels to have some semblance of order
    cc = km.cluster_centers_.flatten()
    temp = [(cc[i],i) for i in range(3)]
    temp = sorted(temp,key=lambda x: x[0])
    
    labels = np.zeros(len(lb),dtype=int)
    for i in range(1,3):
        old_lb = temp[i][1]
        idx = np.where(lb==old_lb)[0]
        labels[idx] = i
        
    x = np.arange(len(labels))
    y = ema[1:]
        
    return x,y,labels