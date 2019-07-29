#!/usr/bin/env python3

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "scripts"

import os
import ccxt
import json
import seaborn
import warnings
import argparse
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.scrape import scrape_ohlcv
from utils.transforms import boxcox
from utils.cmd import parse_args
from utils.csv import create_csv_files, open_convert_csv_files
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation
from plot.ts import tsplot
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm
from matplotlib.dates import YearLocator, MonthLocator

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.graphics.gofplots import qqplot

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
returns = df.returns


def plot_in_sample_hidden_states(hmm_model, df):
    """
    Plot the adjusted closing prices masked by
    the in-sample hidden states as a mechanism
    to understand the market regimes.
    """
    # Predict the hidden states array
    stacked_returns = np.column_stack([df.returns])
    hidden_states = hmm_model.predict(stacked_returns)
    # Create the correctly formatted plot
    fig, axs = plt.subplots(
        hmm_model.n_components,
        sharex=True, sharey=True
    )

    colours = cm.rainbow(
        np.linspace(0, 1, hmm_model.n_components)
    )

    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax.plot_date(
            df.index[mask],
            df.close[mask],
            ".", linestyle='none',
            c=colour
        )
        ax.set_title("Hidden State #%s" % i)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()

stacked_returns = np.column_stack([df.returns])
print(stacked_returns)
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(stacked_returns)
print('Model Score:', hmm_model.score(stacked_returns))
plot_in_sample_hidden_states(hmm_model, df)


# tsplot(returns, lags=30, title='Time Series Analysis Plot')




