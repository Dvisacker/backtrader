#!/usr/bin/env python3
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "scripts"

import os
import ccxt
import json
import numpy as np
import warnings
import argparse
import scipy.stats as stats
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.scrape import scrape_ohlcv
from utils.bars import BAR_TYPES
from utils.csv import create_csv_files, open_convert_csv_files
from utils.cmd import default_parser
from utils.transforms import boxcox
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.graphics.gofplots import qqplot

warnings.filterwarnings("ignore")


configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)

parser = default_parser()

parser.add_argument('-bt', '--bar_type',
                    type=str,
                    default='time',
                    choices=['time', 'tick', 'volume', 'base_volume', 'quote_volume'])

args = parser.parse_args()

exchange_name = args.exchange or default_settings['default_exchange']
start = args.from_date or default_settings['default_start_date']
end = args.to_date or default_settings['default_end_date']
timeframe = args.timeframe or default_settings['default_timeframe']
bar_type = BAR_TYPES[args.bar_type]
symbol = args.symbols[0]

print('Processing time bars')
time_bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type='time_bars')
print('Processing tick bars')
tick_bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type='tick_bars')
print('Processing contract volume bars')
contract_volume_bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type='contract_volume_bars')
print('Process base currency volume bars')
base_currency_volume_bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type='base_volume_bars')
print('Process quote currency volume bars')
quote_currency_volume_bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type='quote_volume_bars')

plt.figure(figsize = (15, 8))
plt.hist(time_bars.close.pct_change().dropna().values.tolist(),
        label = 'Time bars', alpha = 0.5, normed=True, bins=20,
        range = (-0.01, 0.01))

plt.legend()

plt.figure(figsize = (15, 8))
plt.hist(tick_bars.close.pct_change().dropna().values.tolist(),
        label = 'Tick bars', alpha = 0.5, normed=True, bins=20,
        range = (-0.01, 0.01))

plt.legend()

plt.figure(figsize = (15, 8))
plt.hist(contract_volume_bars.close.pct_change().dropna().values.tolist(),
        label = 'Volume bars', alpha = 0.5, normed=True, bins=20,
        range = (-0.01, 0.01))

plt.legend()

plt.figure(figsize = (15, 8))
plt.hist(base_currency_volume_bars.close.pct_change().dropna().values.tolist(),
        label = 'Base Currency Bars', alpha = 0.5, normed=True, bins=20,
        range = (-0.01, 0.01))

plt.legend()

plt.figure(figsize = (15, 8))
plt.hist(quote_currency_volume_bars.close.pct_change().dropna().values.tolist(),
        label = 'Quote Currency Bars', alpha = 0.5, normed=True, bins=20,
        range = (-0.01, 0.01))

plt.legend()

plt.figure(figsize = (15, 8))
plt.hist(time_bars.close.pct_change().dropna().values.tolist(),
        label = 'Time bars', alpha = 0.4, normed=True, bins=20,
        range = (-0.01, 0.01))

plt.hist(tick_bars.close.pct_change().dropna().values.tolist(),
        label = 'Tick bars', alpha = 0.5, normed=True, bins=20,
        range = (-0.01, 0.01))

plt.hist(contract_volume_bars.close.pct_change().dropna().values.tolist(),
        label = 'Contract Volume bars', alpha = 0.4, normed=True, bins=20,
        range = (-0.01, 0.01))

plt.hist(base_currency_volume_bars.close.pct_change().dropna().values.tolist(),
        label = 'Base Currency Volume Bars', alpha = 0.4, normed=True, bins=20,
        range = (-0.01, 0.01))

plt.hist(quote_currency_volume_bars.close.pct_change().dropna().values.tolist(),
        label = 'Quote Currency Volume Bars', alpha = 0.4, normed=True, bins=20,
        range = (-0.01, 0.01))

plt.legend()


print('-' * 20)
print('-' * 20)
print("AUTOCORRELATIONS")
print('Time bars: ', pd.Series.autocorr(time_bars.close.pct_change().dropna()))
print('Tick bars: ',pd.Series.autocorr(tick_bars.close.pct_change().dropna()))
print('Contract Volume bars: ',pd.Series.autocorr(contract_volume_bars.close.pct_change().dropna()))
print('Base Volume bars: ',pd.Series.autocorr(base_currency_volume_bars.close.pct_change().dropna()))
print('Quote Volume bars: ',pd.Series.autocorr(quote_currency_volume_bars.close.pct_change().dropna()))

print('-' * 20)
print('-' * 20)
print("VARIANCE")
print('Time bars: ', np.var(time_bars.close.pct_change().dropna()))
print('Tick bars: ',np.var(tick_bars.close.pct_change().dropna()))
print('Contract Volume bars: ',np.var(contract_volume_bars.close.pct_change().dropna()))
print('Base Volume bars: ',np.var(base_currency_volume_bars.close.pct_change().dropna()))
print('Quote Volume bars: ',np.var(quote_currency_volume_bars.close.pct_change().dropna()))

print('-' * 20)
print('-' * 20)
print("JARQUE BERA TEST")
print('Time bars: ', stats.jarque_bera(time_bars.close.pct_change().dropna()))
print('Tick bars: ', stats.jarque_bera(tick_bars.close.pct_change().dropna()))
print('Contract Volume bars: ', stats.jarque_bera(contract_volume_bars.close.pct_change().dropna()))
print('Base Volume bars: ', stats.jarque_bera(base_currency_volume_bars.close.pct_change().dropna()))
print('Quote Volume bars: ', stats.jarque_bera(quote_currency_volume_bars.close.pct_change().dropna()))

plt.show()
