#!/usr/bin/env python3
# based on http://www.blackarbs.com/blog/mixture-model-trading-part-1/1/16/2018

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
import scipy.stats as stats
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.scrape import scrape_ohlcv
from utils.csv import create_csv_files, open_convert_csv_files
from utils.cmd import parse_args
from utils.transforms import boxcox
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

def add_mean_std_text(x, **kwargs):
    """fn: add mean, std text to seaborn plot

    # Args
        x : pd.Series()
    """
    mean, std = x.mean(), x.std()
    mean_txt = f"mean: {mean:.4%}\nstd: {std:.4%}"

    options = dict(size=10, fontweight='demi', color='red', rotation=0)
    ymin, ymax = plt.gca().get_ylim()
    plt.text(mean+0.025, 0.8*ymax, mean_txt, **options)
    return

def plot_dist(rs, name):
    """fn: to plot single distro with fitted histograms using FacetGrid

    # Args
        rs : pd.DataFrame(), return df
        name : str(), security/column name
    """
    g = (rs
         .pipe(sns.FacetGrid, size=5, aspect=1.5)
         .map(sns.distplot, name, kde=False, fit=stats.norm, fit_kws={'lw':2, 'color': 'blue', 'label':'Normal Distribution'})
         .map(sns.distplot, name, kde=False, fit=stats.laplace, fit_kws={'linestyle':'--', 'color': 'red', 'lw':2, 'label':'Laplace Distribution'})
         .map(sns.distplot, name, kde=False, fit=stats.johnsonsu, fit_kws={'linestyle':'-', 'color': 'green', 'lw':2, 'label':'Johnson SU Distribution'})
         .map(add_mean_std_text, name))
    g.add_legend()
    sns.despine(offset=1)
    plt.title(f'{name}')
    return

def plot_monthly_distributions(x, title):
    """fn: to plot multiple fitted histograms using FacetGrid

    # Args
        x : pd.DataFrame(), return df
        ex : str(), security/column name
    """
    plt.rcParams['font.size'] = 10
    df = x.assign(month=lambda df: df.index.month)
    g = (sns.FacetGrid(df, col='month', col_wrap=2, size=6, aspect=1.2) # make sure to add legend
         .map(sns.distplot, title, kde=False, fit=stats.norm, fit_kws={'color':'blue', 'lw':1.0, 'label':'Normal Distribution'})
         .map(sns.distplot, title, kde=False, fit=stats.laplace, fit_kws={'linestyle':'--', 'color':'red', 'lw':1.0, 'label':'Laplace Distribution'})
         .map(sns.distplot, title, kde=False, fit=stats.johnsonsu, fit_kws={'linestyle':'-', 'color':'green', 'lw':1.0, 'label':'Johnson SU Distribution'})
         .map(add_mean_std_text, title))

    g.add_legend()
    g.fig.subplots_adjust(hspace=.20)
    sns.despine(offset=1)

    df.groupby('month')[title].agg(['mean', 'std']).plot(marker='o', subplots=True)
    return

def plot_weekly_distributions(x, title):
    """fn: to plot multiple fitted histograms using FacetGrid

    # Args
        x : pd.DataFrame(), return df
        ex : str(), security/column name
    """
    plt.rcParams['font.size'] = 10
    df = x.assign(week=lambda df: df.index.week)
    g = (sns.FacetGrid(df, col='week', col_wrap=2, size=6, aspect=1.2) # make sure to add legend
         .map(sns.distplot, title, kde=False, fit=stats.norm, fit_kws={'color':'blue', 'lw':1.0, 'label':'Normal Distribution'})
         .map(sns.distplot, title, kde=False, fit=stats.laplace, fit_kws={'linestyle':'--', 'color':'red', 'lw':1.0, 'label':'Laplace Distribution'})
         .map(sns.distplot, title, kde=False, fit=stats.johnsonsu, fit_kws={'linestyle':'-', 'color':'green', 'lw':1.0, 'label':'Johnson SU Distribution'})
         .map(add_mean_std_text, title))

    g.add_legend()
    g.fig.subplots_adjust(hspace=.20)
    sns.despine(offset=1)

    df.groupby('week')[title].agg(['mean', 'std']).plot(marker='o', subplots=True)
    return



plot_monthly_distributions(df, 'returns')
plt.show()