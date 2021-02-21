import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from sklearn.cluster import KMeans
from datetime import datetime,timedelta

from scipy.stats import gaussian_kde,norm,t
from arch import arch_model


from get_dists import *
from get_trends import *
from get_volatility import *
from hma import *
from plot_functions import *
from plot_trends import *
from trend_prob import *

def get_data(ticker):
    ticker = ticker.upper()
    end_date = datetime.today()
    start_date = end_date-timedelta(days=1000) #arbitrary number of days
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    df = web.DataReader(ticker,'yahoo',start_date,end_date) #type=pandas DataFrame
    return df
#stock data
stock = get_data('tqqq') #Input stock ticker here
stock.Close.plot()

#smoothing of data using forward-backward pass EMA 
c = stock.Close.values
x,y,lb = get_trends(c,20)
yy = c[1:]

plot_trends(x,y,lb)

plot_trends(x,yy,lb)

#log returns
lr = np.diff(np.log(c)) 

#Get Data
volume = stock.Volume.values[1:]
volatility = get_volatility(lr) #Must be rescaled to use in MC like standard dev.
dist_params = get_dists(lr,lb)

plot_vol(volume,lb)
#...
plot_vol(volatility,lb)
#...
plot_dists(*dist_params)

#Hull moving average smoothing
hh = hma(c,20)
hlr = np.diff(np.log(hh.values))
dist_params2 = get_dists(hlr,lb)
plot_dists(*dist_params2)

avg,pcts = get_trend_probs(lb)
plot_trend_probs(pcts)





