#!/usr/bin/env python3

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "scripts"

import os
import ccxt
import pdb
import json
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.scrape import scrape_ohlcv
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation
from sklearn.linear_model import LinearRegression

csv_dir = 'data'
exchange_name = 'bitmex'

instruments = {
  "ETH/BTC", "EOS/BTC", "XRP/BTC"
}

target_instrument = {
  "EOS/BTC"
}

class LinearRegressionModel(object):
    def __init__(self):
        self.df_result = pd.DataFrame(columns=['Actual', 'Predicted'])

    def get_model(self):
        return LinearRegression(fit_intercept=False)

    def learn(self, df, ys, start_date, end_date, lookback_period=20):
        model = self.get_model()

        for date in df[start_date:end_date].index:
            # Fit the model
            x = self.get_prices_since(df, date, lookback_period)
            y = self.get_prices_since(ys, date, lookback_period)
            model.fit(x, y.ravel())

            # Predict the current period
            x_current = df.loc[date].values
            [y_pred] = model.predict([x_current])

            # Store predictions
            new_index = pd.to_datetime(date, format='%Y-%m-%d')
            y_actual = ys.loc[date]
            self.df_result.loc[new_index] = [y_actual, y_pred]

    def get_prices_since(self, df, date_since, lookback):
        index = df.index.get_loc(date_since)
        return df.iloc[index-lookback:index]

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


configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)

  exchange_name = 'bitmex'
  start = default_settings['default_start_date']
  end = default_settings['default_end_date']
  timeframe = '1m'

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

create_csv_files(exchange_name, instruments, timeframe, start, end)
lagged_returns = pd.DataFrame()

for s in instruments:
  df = open_convert_csv_files(exchange_name, s, timeframe, start, end)
  df_close = df['close']

  df_returns = df_close.pct_change(periods=1)
  df_returns_5 = df_close.pct_change(periods=5)
  df_returns_15 = df_close.pct_change(periods=15)
  df_returns_60 = df_close.pct_change(periods=60)
  lagged_returns['%s_1m' % s] = df_close.pct_change(periods=1)
  lagged_returns['%s_5m' % s] = df_close.pct_change(periods=5)
  lagged_returns['%s_15m' % s] = df_close.pct_change(periods=15)
  lagged_returns['%s_60m' % s] = df_close.pct_change(periods=60)


target_prices = open_convert_csv_files(exchange_name, s, timeframe, start, end)
target_returns = target_prices['close'].pct_change().dropna()
start = lagged_returns.index.values[0]
end = lagged_returns.index.values[-1]

multi_linear_model = LinearRegressionModel()
multi_linear_model.learn(lagged_returns, target_returns, start_date=start, end_date=end, lookback_period=10)
multi_linear_model.df_result.plot(
  title='Results',
  style=['-', '--'],
  figsize=(15,10)
)



