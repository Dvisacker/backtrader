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
import matplotlib.pyplot as plt

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.scrape import scrape_ohlcv
from utils.csv import create_csv_files, open_convert_csv_files
from utils.cmd import parse_args
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



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
df = open_convert_csv_files(exchange_name, args.symbols, timeframe, start, end)


def plot_correlograms(X):
  plt.style.use('ggplot')
  plt.rcParams['axes.grid'] = True

  df['z_close'] = (df['close'] - df.close.rolling(window=12).mean()) / df.close.rolling(window=12).std()
  df['zp_close'] = df['z_close'] - df['z_close'].shift(12)

  fig, ax = plt.subplots(4, figsize=(15,10))
  plot_acf(df.z_close.dropna(), ax=ax[0], lags=20)
  plot_pacf(df.z_close.dropna(), ax=ax[1], lags=20)
  plot_acf(df.returns.dropna(), ax=ax[2], lags=20)
  plot_pacf(df.returns.dropna(), ax=ax[3], lags=20)
  ax[0].title.set_text('Detrended prices autocorrelation')
  ax[1].title.set_text('Detrended prices partial autocorrelation')
  ax[2].title.set_text('Returns autocorrelation')
  ax[3].title.set_text('Returns partial autocorrelation')

  plt.tight_layout()
  plt.show()



plot_correlograms(df)