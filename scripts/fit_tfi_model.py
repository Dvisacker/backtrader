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
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.graphics.gofplots import qqplot

warnings.filterwarnings("ignore")


configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)

parser = default_parser()

args = parser.parse_args()

exchange_name = args.exchange or default_settings['default_exchange']
start = args.from_date or default_settings['default_start_date']
end = args.to_date or default_settings['default_end_date']
timeframe = args.timeframe or default_settings['default_timeframe']
symbol = args.symbols[0]

print('Processing time bars')
bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type='flow_imbalance_bars')
print('Processing tick bars')

def fit_next_returns_to_tfi(df, lag=1):
    df['next_midprice_returns'] = df.midprice_returns.shift(lag)

    df.dropna(inplace=True)
    df.plot(kind='scatter', grid=True,
                            x='tfi', y='next_midprice_returns',
                            title = 'Returns/TFI Correlation',
                            alpha=0.5, figsize=(12,10))


    tfi_ = sm.add_constant(df['tfi'])
    ols = sm.OLS(df.next_midprice_returns, tfi_).fit()
    print(ols.summary2())
    plt.show()


fit_next_returns_to_tfi(bars)