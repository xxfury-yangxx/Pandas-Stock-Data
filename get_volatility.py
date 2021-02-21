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

#Volatility using GARCH(1,1)
def get_volatility(lr):
    gmodel = arch_model(
        lr*100,
        vol='GARCH',
        mean='Zero',
        p=1,
        q=1,
        dist='StudentsT'
        )
    
    gfit = gmodel.fit(update_freq=5)
    vol = gfit.conditional_volatility
    return vol