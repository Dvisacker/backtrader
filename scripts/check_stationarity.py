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

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.scrape import scrape_ohlcv
from utils.csv import create_csv_files, open_convert_csv_files
from utils.cmd import parse_args
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation
from statsmodels.tsa.stattools import adfuller, coint

args = parse_args()

configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)


exchange_name = args.exchange or default_settings['default_exchange']
start = args.from_date or default_settings['default_start_date']
end = args.to_date or default_settings['default_end_date']
timeframe = args.timeframe or default_settings['default_timeframe']


symbol1 = from_standard_to_exchange_notation(exchange_name, args.symbols[0], index=True)
symbols = [symbol1]

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
if symbol1 not in exchange.symbols:
    print('-'*36,' ERROR ','-'*35)
    print('The requested symbol is not available from {}\n'.format(exchange_name))
    print('Available symbols are:')
    print('-'*80)
    quit()


create_csv_files(exchange_name, [args.symbols[0]], timeframe, start, end)
df = open_convert_csv_files(exchange_name, args.symbols[0], timeframe, start, end)


def check_for_stationarity(X):
  df['z_close'] = (df['close'] - df.close.rolling(window=12).mean()) / df.close.rolling(window=12).std()
  df['zp_close'] = df['z_close'] - df['z_close'].shift(12)

  print("\n> Is the prices series stationary ?")
  dftest = adfuller(df.close, autolag='AIC')
  print("Test statistic = {:.3f}".format(dftest[0]))
  print("P-value = {:.3f}".format(dftest[1]))
  print("Critical values :")
  for k, v in dftest[4].items():
      print("\t{}: {} - The price series is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))

  print("\n> Is the returns series stationary ?")
  dftest = adfuller(df.returns, autolag='AIC')
  print("Test statistic = {:.3f}".format(dftest[0]))
  print("P-value = {:.3f}".format(dftest[1]))
  print("Critical values :")
  for k, v in dftest[4].items():
      print("\t{}: {} - The returns series is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))

  print("\n> Is the de-trended price series stationary ?")
  dftest = adfuller(df.z_close.dropna(), autolag='AIC')
  print("Test statistic = {:.3f}".format(dftest[0]))
  print("P-value = {:.3f}".format(dftest[1]))
  print("Critical values :")
  for k, v in dftest[4].items():
      print("\t{}: {} - The price series is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))

  print("\n> Is the 12-lag differenced de-trended price series stationary ?")
  dftest = adfuller(df.zp_close.dropna(), autolag='AIC')
  print("Test statistic = {:.3f}".format(dftest[0]))
  print("P-value = {:.3f}".format(dftest[1]))
  print("Critical values :")
  for k, v in dftest[4].items():
      print("\t{}: {} - The price series is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))

  print('\n\n')

check_for_stationarity(df)