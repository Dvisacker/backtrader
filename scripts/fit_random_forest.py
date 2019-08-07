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
import statsmodels.api as sm

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
from models import all_models

warnings.filterwarnings("ignore")

configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)

parser = default_parser()

parser.add_argument('-bt', '--bar_type',
                    type=str,
                    choices=list(BAR_TYPES.keys()))

parser.add_argument('-m', '--model',
                     type=str,
                     required=True,
                     choices=list(all_models.keys()))

parser.add_argument('-ic', '--include_correlations',
                    type=str,
                    nargs='+',
                    required=False,
                    help='Correlations you want to include')


args = parser.parse_args()

exchange_name = args.exchange or default_settings['default_exchange']
start = args.from_date or default_settings['default_start_date']
end = args.to_date or default_settings['default_end_date']
timeframe = args.timeframe or default_settings['default_timeframe']
symbol = args.symbols[0]
included_correlation_symbols  = args.include_correlations
bar_type = BAR_TYPES[args.bar_type]
create_model = all_models[args.model]

bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type=bar_type)

raw_features = {}
if included_correlation_symbols:
  for s in included_correlation_symbols:
    raw_features[s] = open_convert_csv_files(exchange_name, s, timeframe, start, end, bar_type=bar_type)

create_model(bars, raw_features)