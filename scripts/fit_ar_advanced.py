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
import pandas as pd
import numpy as np

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.scrape import scrape_ohlcv
from utils.csv import create_csv_files, open_convert_csv_files
from utils.cmd import parse_args
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation

from scipy import stats
from statsmodels.stats.stattools import jarque_bera
import statsmodels.api as sm
import statsmodels.tsa as tsa
import matplotlib.pyplot as plt

def plot_acf(X_acf, X_acf_confs, title='ACF'):
    # The confidence intervals are returned by the functions as (lower, upper)
    # The plotting function needs them in the form (x-lower, upper-x)
    errorbars = np.ndarray((2, len(X_acf)))
    errorbars[0, :] = X_acf - X_acf_confs[:,0]
    errorbars[1, :] = X_acf_confs[:,1] - X_acf

    plt.plot(X_acf, 'ro')
    plt.errorbar(range(len(X_acf)), X_acf, yerr=errorbars, fmt='none', ecolor='gray', capthick=2)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(title)


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

N = 10
AIC = np.zeros((N, 1))

for i in range(N):
  model = tsa.api.AR(returns, freq=pd.offsets.Minute(5))
  model = model.fit(maxlag=(i+1))
  AIC[i] = model.aic

AIC_min = np.min(AIC)
AIC_model_min = np.argmin(AIC)

print('Relative Likelyhoods')
print('Number of parameters in minimum AIC model %s' % (AIC_model_min + 1))
# print(np.exp((AIC_min - AIC) / 2))


BIC = np.zeros((N, 1))

for i in range(N):
  model = tsa.api.AR(returns, freq=pd.offsets.Minute(5))
  model = model.fit(maxlag=(i+1))
  BIC[i] = model.bic


BIC_min = np.min(BIC)
BIC_model_min = np.argmin(BIC)

print('Relative Likelyhoods')
print('Number of parameters in minimum BIC model %s' % (BIC_model_min + 1))
# print(np.exp((BIC_min - BIC) / 2))


model_min = min(BIC_model_min, AIC_model_min)
final_model = tsa.api.AR(returns, freq=pd.offsets.Minute(5))
final_model = final_model.fit(maxlag=model_min)

score, pvalue, _, _ = jarque_bera(model.resid)

if pvalue < 0.10:
  print("The model residuals don't seem to be normally distributed")
else:
  print("The model residuals seem to be normally distributed")

final_model_confs = np.asarray((final_model.params - final_model.bse, final_model.params + final_model.bse)).T
plot_acf(final_model.params, final_model_confs, title='Model Estimated Parameters')
plt.show()

print('Parameters')
print(final_model.params)
print('Standard Error')
print(final_model.bse)