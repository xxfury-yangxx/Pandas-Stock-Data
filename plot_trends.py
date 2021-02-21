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

def plot_trends(x,y,lb):
    clist = ['r','orange','g']
    nclasses = len(np.unique(lb))
    for i in range(nclasses):
        xx = x[lb==i]
        yy = y[lb==i]
        plt.scatter(xx,yy,c=clist[i],label=str(i))
    plt.legend(fontsize='x-large')
    plt.show()
    
