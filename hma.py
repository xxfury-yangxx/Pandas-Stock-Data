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

def hma(c,w):
    cs = pd.Series(c)
    ema1 = 2*cs.ewm(span=w//2).mean()-cs.ewm(span=w).mean()
    hma = ema1.ewm(span=int(np.sqrt(w))).mean()
    return hma