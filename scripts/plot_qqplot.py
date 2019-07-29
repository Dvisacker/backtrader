#!/usr/bin/env python3

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "scripts"

import os
import ccxt
import json
import warnings
import argparse
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.scrape import scrape_ohlcv
from utils.transforms import boxcox
from utils.csv import create_csv_files, open_convert_csv_files
from utils.cmd import parse_args
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.graphics.gofplots import qqplot

configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)

args = parse_args()
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
returns = df.returns

plt.style.use('ggplot')
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 1.5
fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(3,1, width_ratios=[1], height_ratios=[1,1,1], hspace=0.6, wspace=0.4)

returns_ax = fig.add_subplot(gs[0], ylabel='Returns, %')
returns_ax.xaxis.label.set_visible(False)
hist_ax = fig.add_subplot(gs[1], ylabel='Histogram')
qq_ax = fig.add_subplot(gs[2], ylabel='QQ Plot')

returns.plot(ax=returns_ax, color='orange', lw=1.5)
hist_ax.hist(returns, bins=200)
qqplot(returns, line='r', ax=qq_ax, lw=1., markersize=2)

plt.show()



