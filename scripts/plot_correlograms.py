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
from utils.bars import BAR_TYPES
from utils.csv import create_csv_files, open_convert_csv_files
from utils.cmd import default_parser
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)


parser = default_parser()

parser.add_argument('-bt', '--bar_type',
                    type=str,
                    default='time',
                    choices=['time', 'tick', 'volume', 'base_volume', 'quote_volume']
)

args = parser.parse_args()

exchange_name = args.exchange or default_settings['default_exchange']
start = args.from_date or default_settings['default_start_date']
end = args.to_date or default_settings['default_end_date']
timeframe = args.timeframe or default_settings['default_timeframe']
bar_type = BAR_TYPES[args.bar_type]
symbol = args.symbols[0]

bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type=bar_type)
returns = bars.returns
plot_acf(returns, lags=20)

# symbol = from_standard_to_exchange_notation(exchange_name, args.symbols, index=True)
# symbols = [symbol]
# create_csv_files(exchange_name, [args.symbols], timeframe, start, end)
# df = open_convert_csv_files(exchange_name, args.symbols, timeframe, start, end)
# def plot_correlograms():
# def plot_correlograms(X):
#   plt.style.use('ggplot')
#   plt.rcParams['axes.grid'] = True

#   df['z_close'] = (df['close'] - df.close.rolling(window=12).mean()) / df.close.rolling(window=12).std()
#   df['zp_close'] = df['z_close'] - df['z_close'].shift(12)

#   fig, ax = plt.subplots(4, figsize=(15,10))
#   plot_acf(df.z_close.dropna(), ax=ax[0], lags=20)
#   plot_pacf(df.z_close.dropna(), ax=ax[1], lags=20)
#   plot_acf(df.returns.dropna(), ax=ax[2], lags=20)
#   plot_pacf(df.returns.dropna(), ax=ax[3], lags=20)
#   ax[0].title.set_text('Detrended prices autocorrelation')
#   ax[1].title.set_text('Detrended prices partial autocorrelation')
#   ax[2].title.set_text('Returns autocorrelation')
#   ax[3].title.set_text('Returns partial autocorrelation')

#   plt.tight_layout()
#   plt.show()



# plot_correlograms(df)