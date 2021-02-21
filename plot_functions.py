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

#Plotting Functions
def plot_vol(vol,lb): #used for plotting volume and volatility
    colors = ['r','orange','g']
    for i in range(3):
        temp = vol[lb==i]
        x = np.where(lb==i)[0]
        plt.scatter(x,temp,c=colors[i],label=str(i))
    plt.legend(fontsize='x-large')
    plt.show()

def plot_dists(xr,kde_dict,t_dict,norm_dict): #Used to compare various dists
    colors = ['r','orange','g']
    fig,axes = plt.subplots(nrows=3,sharex=True)
    for i in range(3):
        y1,y2,y3 = kde_dict[str(i)],t_dict[str(i)],norm_dict[str(i)]
        axes[0].plot(xr,y1,c=colors[i],label=str(i))
        axes[1].plot(xr,y2,c=colors[i],label=str(i))
        axes[2].plot(xr,y3,c=colors[i],label=str(i))
    
    plt.legend(fontsize='x-large')
    plt.show()