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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa import ar_model

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.scrape import scrape_ohlcv
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation

csv_dir = 'data'

def _date_parse(timestamp):
        """
        Parses timestamps into python datetime objects.
        """
        return datetime.fromtimestamp(float(timestamp))

def create_csv_files(exchange, symbols, timeframe, start, end):
    for symbol in symbols:
      csv_filename = get_ohlcv_file(exchange, symbol, timeframe, start, end)
      csv_filepath = os.path.join(csv_dir, csv_filename)
      if not os.path.isfile(csv_filepath):
        print('Downloading {}'.format(csv_filename))
        scrape_ohlcv(exchange, symbol, timeframe, start, end)
        print('Downloaded {}'.format(csv_filename))


def open_convert_csv_files(exchange, symbol, timeframe, start, end):
    """
    Opens the CSV files from the data directory, converting
    them into pandas DataFrames within a symbol dictionary.
    For this handler it will be assumed that the data is
    taken from Yahoo. Thus its format will be respected.
    """
    csv_filename = get_ohlcv_file(exchange, symbol, timeframe, start, end)
    csv_filepath = os.path.join(csv_dir, csv_filename)
    df = pd.read_csv(
        csv_filepath,
        parse_dates=True,
        date_parser=_date_parse,
        header=0,
        sep=',',
        index_col=1,
        names=['time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    df['returns'] = df['close'].pct_change()
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    df.dropna(inplace=True)

    data = df.sort_index()
    return data

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



def parse_args():
    parser = argparse.ArgumentParser(description='Market data downloader')


    parser.add_argument('-s','--symbol',
                        type=str,
                        required=True,
                        help='The Symbol of the Instrument/Currency Pair To Download')

    parser.add_argument('-e','--exchange',
                        type=str,
                        help='The exchange to download from')

    parser.add_argument('-t','--timeframe',
                        type=str,
                        default='1m',
                        choices=['1m', '5m','15m', '30m','1h', '2h', '3h', '4h', '6h', '12h', '1d', '1M', '1y'],
                        help='The timeframe to download')

    parser.add_argument('-days', '--days',
                         type=int,
                         help='The number of days to fetch ohlcv'
                        )

    parser.add_argument('-from', '--from_date',
                         type=str,
                         help='The date from which to start dowloading ohlcv from'
                        )

    parser.add_argument('-end', '--to_date',
                         type=str,
                         help='The date up to which to download ohlcv to'
                        )

    parser.add_argument('--debug',
                            action ='store_true',
                            help=('Print Sizer Debugs'))

    return parser.parse_args()



args = parse_args()

configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)


exchange_name = args.exchange or default_settings['default_exchange']
start = args.from_date or default_settings['default_start_date']
end = args.to_date or default_settings['default_end_date']
timeframe = args.timeframe or default_settings['default_timeframe']


symbol = from_standard_to_exchange_notation(exchange_name, args.symbol, index=True)
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


create_csv_files(exchange_name, [args.symbol], timeframe, start, end)
df = open_convert_csv_files(exchange_name, args.symbol, timeframe, start, end)

prices = df['close']
model = ar_model(prices)
model = model.fit()

print('Parameters')
print(model.params)

print('Standard Error')
print(model.bse)

# To plot this we'll need to format a confidence interval 2D array like the previous functions returned
# Here is some quick code to do that
model_confs = np.asarray((model.params - model.bse, model.params + model.bse)).T
plot_acf(model.params, model_confs, title='Model Estimated Parameters')

# We have to set a confidence level for our intervals, we choose the standard of 95%,
# corresponding with an alpha of 0.05.
nlags=10
X_acf, X_acf_confs = acf(prices, nlags=nlags, alpha=0.05)
X_pacf, X_pacf_confs = pacf(prices, nlags=nlags, alpha=0.05)

plot_acf(X_acf, X_acf_confs)