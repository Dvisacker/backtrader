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
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.csv import create_csv_files, open_convert_csv_files
from utils.scrape import scrape_ohlcv
from utils.cmd import parse_args
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation



configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)

args = parse_args()
exchange_name = args.exchange or default_settings['default_exchange']
start = args.from_date or default_settings['default_start_date']
end = args.to_date or default_settings['default_end_date']
timeframe = args.timeframe or default_settings['default_timeframe']


symbol1 = from_standard_to_exchange_notation(exchange_name, args.symbols[0], index=True)
symbol2 = from_standard_to_exchange_notation(exchange_name, args.symbols[1], index=True)
symbols = [symbol1, symbol2]

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

# Check if the symbol is available on the Exchange
exchange.load_markets()
if symbol1 not in exchange.symbols or symbol2 not in exchange.symbols:
    print('-'*36,' ERROR ','-'*35)
    print('The requested symbol is not available from {}\n'.format(exchange_name))
    print('Available symbols are:')
    print('-'*80)
    quit()


create_csv_files(exchange_name, [args.symbols[0], args.symbols[1]], timeframe, start, end)
df1 = open_convert_csv_files(exchange_name, args.symbols[0], timeframe, start, end)
df2 = open_convert_csv_files(exchange_name, args.symbols[1], timeframe, start, end)
window = int(len(df1.index)/2)

returns_1 = df1['returns']
returns_2 = df2['returns']

plt.style.use('ggplot')
plt.rcParams['axes.grid']
plt.rcParams['lines.linewidth'] = 1.5

df_corrs = returns_1 \
        .rolling(window=window, min_periods=window) \
        .corr(other=returns_2) \
        .dropna()


df_corrs.plot(figsize=(20,10))
plt.tight_layout()
plt.show()



