#!/usr/bin/env python3
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "scripts"

import os
import pdb
import json
import argparse
import pandas as pd
from utils.helpers import from_exchange_to_standard_notation, get_bar_data_filename
from utils.scrape import scrape_bitmex_trades
from utils.bars import FlowImbalanceBarSeries, BarSeriesUtils, TIME_FREQUENCIES
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

quote_df = pd.DataFrame()

while current_date < to_date:
    current_date_string = datetime.strftime(current_date, '%Y%m%d')
    daily_quote_df = pd.read_csv('data/quote_data/{}.csv.gz'.format(current_date_string))
    print('Appending ...')
    quote_df = quote_df.append(daily_quote_df)
    current_date += timedelta(days=1)


from_date_string = from_date.strftime('%d/%m/%Y').replace("/", "")
to_date_string = to_date.strftime('%d/%m/%Y').replace("/", "")
filename = 'bitmex-{}-{}.csv'.format(from_date_string, to_date_string).replace("/", "")

quote_data_filepath = os.path.join(data_dir, 'quote_data', filename)
trade_data_filepath = os.path.join(data_dir, 'tick_data', filename)

quote_df.to_csv(quote_data_filepath)
trade_df = pd.read_csv(trade_data_filepath)

print('Concatenated tick data')
symbols = quote_df.symbol.unique()
time_frequency=TIME_FREQUENCIES[timeframe]
frequencies = list(TIME_FREQUENCIES.keys())

for s in symbols:
  print('Converting {} bars'.format(s))
  current_quote_df = quote_df[quote_df.symbol == s]
  current_quote_df['time'] = current_quote_df.timestamp.map(lambda t: datetime.strptime(t[:-3], "%Y-%m-%dD%H:%M:%S.%f"))
  current_quote_df.set_index('time', inplace=True)

  current_trade_df = trade_df[trade_df.symbol == s]
  current_trade_df['time'] = current_trade_df.timestamp.map(lambda t: datetime.strptime(t[:-3], "%Y-%m-%dD%H:%M:%S.%f"))
  current_trade_df.set_index('time', inplace=True)

  for fr in ["1m", "5m", "15m"]:
    parsed = {
      "1m": "1T",
      "5m": "5T",
      "15m": "15T"
    }[fr]

    time_bars = FlowImbalanceBarSeries(current_quote_df, current_trade_df).process_ticks(frequency=parsed)
    num_time_bars = time_bars.shape[0]

    standard_symbol = from_exchange_to_standard_notation('bitmex', s)
    filename = get_bar_data_filename('bitmex', standard_symbol, fr, from_date_string, to_date_string)
    filepath = '{}/flow_imbalance_bars/{}'.format(data_dir, filename)
    time_bars.to_csv(filepath)













