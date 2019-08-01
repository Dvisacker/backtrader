#!/usr/bin/env python3
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "scripts"

import os
import json
import argparse
import pandas as pd
from utils.helpers import from_exchange_to_standard_notation
from utils.scrape import scrape_bitmex_trades
from utils.bars import TIME_FREQUENCIES, get_candle_nb, BarSeriesUtils, BarSeries, VolumeBarSeries, BaseCurrencyVolumeBarSeries, QuoteCurrencyVolumeBarSeries, TickBarSeries
from datetime import datetime, timedelta


configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)


parser = argparse.ArgumentParser(description='Market data downloader')
parser.add_argument('-from', '--from_date',
                      type=str,
                      help='The date from which to start dowloading ohlcv from'
                    )

parser.add_argument('-to', '--to_date',
                      type=str,
                      help='The date up to which to download ohlcv to'
                    )

parser.add_argument('-t', '--timeframe',
                     type=str,
                      help='The desired approximated timeframe')

args = parser.parse_args()


from_date = datetime.strptime(args.from_date, '%d/%m/%Y')
to_date = datetime.strptime(args.to_date, '%d/%m/%Y')
data_dir = default_settings['csv_dir']
current_date = from_date
timeframe = args.timeframe or default_settings['default_timeframe']
days = (to_date - from_date).days

df = pd.DataFrame()

while current_date < to_date:
    current_date_string = datetime.strftime(current_date, '%Y%m%d')
    filename = 'data/tick_data/{}.csv.gz'.format(current_date_string)
    daily_df = pd.read_csv(filename)
    print('Appending ...')
    df = df.append(daily_df)
    current_date += timedelta(days=1)

from_date_string = from_date.strftime('%d/%m/%Y').replace("/", "")
to_date_string = to_date.strftime('%d/%m/%Y').replace("/", "")
tick_data_filename = 'bitmex-{}-{}.csv'.format(from_date_string, to_date_string).replace("/", "")
tick_data_filepath = os.path.join(data_dir, 'tick_data', tick_data_filename)
df.to_csv(tick_data_filepath)

print('Concatenated tick data')

symbols = df.symbol.unique()
symbol_dfs = {}

time_frequency=TIME_FREQUENCIES[timeframe]
bar_utils = BarSeriesUtils(days, timeframe)


for s in symbols:
  print('Converting {} bars'.format(s))
  symbol_dfs[s] = df[df.symbol == s]
  symbol_dfs[s]['time'] = symbol_dfs[s].timestamp.map(lambda t: datetime.strptime(t[:-3], "%Y-%m-%dD%H:%M:%S.%f"))
  symbol_dfs[s].set_index('time', inplace=True)

  tick_frequency = bar_utils.get_recommended_tick_frequency(symbol_dfs[s])
  if (tick_frequency == 0):
    print('Tick frequency is equal to 0 for {}'.format(s))
    continue

  contract_volume_frequency = bar_utils.get_recommended_volume_frequency(symbol_dfs[s])
  base_currency_volume_frequency = bar_utils.get_recommended_base_currency_volume_frequency(symbol_dfs[s])
  quote_currency_volume_frequency = bar_utils.get_recommended_quote_currency_volume_frequency(symbol_dfs[s])

  print('Processing time bars ..')
  time_bars = BarSeries(symbol_dfs[s]).process_ticks(frequency=time_frequency)
  print('Processing tick bars ..')
  tick_bars = TickBarSeries(symbol_dfs[s]).process_ticks(frequency=tick_frequency)
  print('Processing contract volume bars ..')
  contract_volume_bars = VolumeBarSeries(symbol_dfs[s]).process_ticks(frequency=contract_volume_frequency)
  print('Processing base volume bars ..')
  base_volume_bars = BaseCurrencyVolumeBarSeries(symbol_dfs[s]).process_ticks(frequency=base_currency_volume_frequency)
  print('Processing quote volume bars ..')
  quote_volume_bars = QuoteCurrencyVolumeBarSeries(symbol_dfs[s]).process_ticks(frequency=quote_currency_volume_frequency)

  num_time_bars = time_bars.shape[0]
  num_contract_volume_bars = contract_volume_bars.shape[0]
  num_tick_bars = tick_bars.shape[0]
  num_base_volume_bars = base_volume_bars.shape[0]
  num_quote_volume_bars = quote_volume_bars.shape[0]

  print('Number of bars for {}'.format(s))
  print(num_time_bars)
  print(num_tick_bars)
  print(num_contract_volume_bars)
  print(num_base_volume_bars)
  print(num_quote_volume_bars)

  general_symbol = from_exchange_to_standard_notation('bitmex', s)
  time_bars_filename = '{}/time_bars/bitmex-{}-{}-{}.csv'.format(data_dir, s, from_date_string, to_date_string)
  tick_bars_filename = '{}/tick_bars/bitmex-{}-{}-{}.csv'.format(data_dir, s, from_date_string, to_date_string)
  contract_volume_bars_filename = '{}/contract_volume_bars/bitmex-{}-{}-{}.csv'.format(data_dir, s, from_date_string, to_date_string)
  base_volume_bars_filename = '{}/base_volume_bars/bitmex-{}-{}-{}.csv'.format(data_dir, s, from_date_string, to_date_string)
  quote_volume_bars_filename = '{}/quote_volume_bars/bitmex-{}-{}-{}.csv'.format(data_dir, s, from_date_string, to_date_string)

  time_bars.to_csv(time_bars_filename)
  tick_bars.to_csv(tick_bars_filename)
  contract_volume_bars.to_csv(contract_volume_bars_filename)
  base_volume_bars.to_csv(base_volume_bars_filename)
  quote_volume_bars.to_csv(quote_volume_bars_filename)














