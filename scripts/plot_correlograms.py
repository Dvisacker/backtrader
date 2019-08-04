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
from utils.helpers import get_ohlcv_file, get_timeframe, move_figure
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

print('Processing time bars')
time_bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type='time_bars')
time_bars_returns = time_bars.returns
print('Processing tick bars')
tick_bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type='tick_bars')
tick_bars_returns = tick_bars.returns
print('Processing contract volume bars')
contract_volume_bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type='contract_volume_bars')
contract_volume_bars_returns = contract_volume_bars.returns
print('Process base currency volume bars')
base_volume_bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type='base_volume_bars')
base_volume_bars_returns = base_volume_bars.returns
print('Process quote currency volume bars')
quote_volume_bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type='quote_volume_bars')
quote_volume_bas_returns = quote_volume_bars.returns

plt.style.use('ggplot')
plt.rcParams['axes.grid'] = True

fig, ax = plt.subplots(3, 2, figsize=(10,10), sharex=True, sharey=True)
plot_acf(time_bars_returns, ax=ax[0,0], lags=20, zero=False)
plot_acf(tick_bars_returns, ax=ax[1,0], lags=20, zero=False)
plot_acf(contract_volume_bars_returns, ax=ax[2,0], lags=20, zero=False)
plot_acf(base_volume_bars_returns, ax=ax[0,1], lags=20, zero=False)
plot_acf(base_volume_bars_returns, ax=ax[1,1], lags=20, zero=False)
ax[0,0].title.set_text('Time bar returns (ACF)')
ax[1,0].title.set_text('Tick bar returns (ACF)')
ax[2,0].title.set_text('Contract volume bar returns (ACF)')
ax[0,1].title.set_text('Base volume bar returns (ACF)')
ax[1,1].title.set_text('Quote volume bar returns (ACF)')
plt.tight_layout()


fig, ax = plt.subplots(3, 2, figsize=(10,10), sharex=True, sharey=True)
plot_pacf(time_bars_returns, ax=ax[0,0], lags=20, zero=False)
plot_pacf(tick_bars_returns, ax=ax[1,0], lags=20, zero=False)
plot_pacf(contract_volume_bars_returns, ax=ax[2,0], lags=20, zero=False)
plot_pacf(base_volume_bars_returns, ax=ax[0,1], lags=20, zero=False)
plot_pacf(base_volume_bars_returns, ax=ax[1,1], lags=20, zero=False)
ax[0,0].title.set_text('Time bar returns (PACF)')
ax[1,0].title.set_text('Tick bar returns (PACF)')
ax[2,0].title.set_text('Contract volume bar returns (PACF)')
ax[0,1].title.set_text('Base volume bar returns (PACF)')
ax[1,1].title.set_text('Quote volume bar returns (PACF)')
plt.tight_layout()
move_figure(fig, 1000, 0)
plt.show()