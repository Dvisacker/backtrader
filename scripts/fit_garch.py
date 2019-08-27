#!/usr/bin/env python3

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "scripts"

import os
import ccxt
import json
import argparse
import math
import pandas as pd
import numpy as np
import warnings

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.scrape import scrape_ohlcv
from utils.cmd import parse_args
from utils.csv import create_csv_files, open_convert_csv_files
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation
from plot.ts import tsplot

from scipy import stats
from statsmodels.stats.stattools import jarque_bera
import statsmodels.api as sm
import statsmodels.tsa as tsa
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from arch import arch_model

warnings.filterwarnings("ignore")



configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)


args = parse_args()
exchange_name = args.exchange or default_settings['default_exchange']
start = args.from_date or default_settings['default_start_date']
end = args.to_date or default_settings['default_end_date']
timeframe = args.timeframe or default_settings['default_timeframe']


symbol = from_standard_to_exchange_notation(exchange_name, args.symbols, index=True)
symbols = [symbol]

# Get our Exchange
try:
    exchange = getattr (ccxt, exchange_name) ()
except AttributeError:
    print('-'*36,' ERROR ','-'*35)
    print('Exchange "{}" not found. Please check the exchange is supported.'.format(exchange_name))
    print('-'*80)
    quit()

# Check if fetching of OHLC Data is supported
if exchange.has["fetchOHLCV"] != True:
    print('-'*36,' ERROR ','-'*35)
    print('{} does not support fetching OHLC data. Please use another exchange'.format(exchange_name))
    print('-'*80)
    quit()

# Check requested timeframe is available. If not return a helpful error.
if (not hasattr(exchange, 'timeframes')) or (timeframe not in exchange.timeframes):
    print('-'*36,' ERROR ','-'*35)
    print('The requested timeframe ({}) is not available from {}\n'.format(timeframe,exchange_name))
    print('Available timeframes are:')
    for key in exchange.timeframes.keys():
        print('  - ' + key)
    print('-'*80)
    quit()

# Check if the symbol is available on the Exchange
exchange.load_markets()
if symbol not in exchange.symbols:
    print('-'*36,' ERROR ','-'*35)
    print('The requested symbol is not available from {}\n'.format(exchange_name))
    print('Available symbols are:')
    print('-'*80)
    quit()


create_csv_files(exchange_name, [args.symbols], timeframe, start, end)
df1 = open_convert_csv_files(exchange_name, args.symbols, timeframe, start, end)
returns = df1['returns']


def _get_best_model(TS):
    best_aic = np.inf
    best_order = None
    best_model = None
    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = sm.tsa.ARIMA(TS, order=(i,d,j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_model = tmp_mdl
                except: continue
    print('aic: {:6.2f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_model


# First fit an ARIMA model
params = _get_best_model(returns)
order = params[1]
model = params[2]

tsplot(model.resid, lags=30, title='Best ARIMA model (Residuals). Order={}'.format(order))
tsplot(model.resid**2, lags=30, title='Best ARIMA model (Residuals Squared). Order={}'.format(order))


# In case we see autocorrelation in squared residuals, we attempt to fit a GARCH model with the best ARIMA model
p_ = order[0]
o_ = order[1]
q_ = order[2]

garch_model = arch_model(model.resid, p=p_, o=o_, q=q_, dist='StudentsT')
garch_model = garch_model.fit(update_freq=5, disp='off')
print(garch_model.summary())

tsplot(garch_model.resid, lags=30, title='Best GARCH Model (Residuals). Order={}'.format(order))
tsplot(garch_model.resid**2, lags=30, title='Best GARCH Model (Residuals Squared). Order={}'.format(order))
plt.show()


