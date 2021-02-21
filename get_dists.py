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

#Code to get KDE of returns and Tdist/Normal dist of returns
def get_dists(lr,lb):
    kde_dict = dict()
    t_dict = dict()
    norm_dict = dict()
    for i in range(3):
        temp = lr[lb==i]
        xr = np.linspace(lr.min(),lr.max(),1000)
        
        kde = gaussian_kde(temp)
        y = kde(xr)
        kde_dict[str(i)] = y
        
        t_pdf = t.pdf(xr,*t.fit(temp))
        n_pdf = norm.pdf(xr,*norm.fit(temp))
        t_dict[str(i)] = t_pdf
        norm_dict[str(i)] = n_pdf
        
    return xr,kde_dict,t_dict,norm_dict
