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

def get_trend_probs(lb):
    avg_tlengths = dict()
    pcts = dict()
    for i in range(3):
        idx = np.where(np.diff(np.where(lb==i)[0])!=1)[0]
        lengths = np.diff(idx)
        m = lengths[lengths>2].mean()
        avg_tlengths[str(i)] = m
        ll = np.unique(lengths)
        
        temp = []
        for x in ll:
            p = len(ll[ll>x])/len(ll)
            temp.append((x,p))
        
        pcts[str(i)] = temp
        
    #Print number of trends; If number is small, get more data!!!
    for i in range(3):
        temp = len(pcts[str(i)])
        print(i,temp)
        
    return avg_tlengths,pcts

def plot_trend_probs(pcts):
    fig,axes = plt.subplots(nrows=3)
    for i in range(3):
        x,y = zip(*pcts[str(i)])
        axes[i].plot(x,y,marker='o')
    plt.show()