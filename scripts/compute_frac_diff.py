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
from models import all_models

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.graphics.gofplots import qqplot

from features.frac import get_optimal_differentiation, compute_fractional_differentiation

warnings.filterwarnings("ignore")
configuration_file = "./scripts/default_settings.json"

with open(configuration_file) as f:
  default_settings = json.load(f)

parser = default_parser()

parser.add_argument('-bt', '--bar_type',
                    type=str,
                    choices=list(BAR_TYPES.keys()))


args = parser.parse_args()

exchange_name = args.exchange or default_settings['default_exchange']
start = args.from_date or default_settings['default_start_date']
end = args.to_date or default_settings['default_end_date']
timeframe = args.timeframe or default_settings['default_timeframe']
symbol = args.symbols[0]
bar_type = BAR_TYPES[args.bar_type]

bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type=bar_type)
prices = bars.close
returns = bars.returns
opt_d = get_optimal_differentiation(prices)
diff = compute_fractional_differentiation(prices, d=opt_d)

print(opt_d)

plt.style.use('ggplot')
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 1.5

fig, ax = plt.subplots(3, figsize=(20,15), sharex=True)
prices.plot(ax=ax[0], color='blue', lw=1., label='{} Prices'.format(symbol))
returns.plot(ax=ax[1], color='red', lw=1., label='{} Returns'.format(symbol))
diff.plot(ax=ax[2], color='orange', lw=1., label='{} Prices Optimal Differentiation'.format(symbol))
plt.show()