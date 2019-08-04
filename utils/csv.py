import os
import pdb
import pandas as pd

from datetime import datetime
from utils.helpers import get_ohlcv_file
from utils.scrape import scrape_ohlcv



def _date_parse(timestamp):
        """
        Parses timestamps into python datetime objects.
        """
        return datetime.fromtimestamp(float(timestamp))


def _parse_datestring(date):
        """
        Parses timestamps into python datetime objects.
        """
        return datetime.strptime(date[:19], '%Y-%m-%d %H:%M:%S')

def create_csv_files(exchange, symbols, timeframe, start, end, csv_dir='data'):
    for symbol in symbols:
      csv_filename = get_ohlcv_file(exchange, symbol, timeframe, start, end)
      bar_type = "time_bars"
      csv_filepath = os.path.join(csv_dir, bar_type, csv_filename)
      if not os.path.isfile(csv_filepath):
        print('Downloading {}'.format(csv_filename))
        scrape_ohlcv(exchange, symbol, timeframe, start, end)
        print('Downloaded {}'.format(csv_filename))


def open_convert_csv_files(exchange, symbol, timeframe, start, end, csv_dir='data', bar_type='time_bars'):
    """
    Opens the CSV files from the data directory, converting
    them into pandas DataFrames within a symbol dictionary.
    For this handler it will be assumed that the data is
    taken from Yahoo. Thus its format will be respected.
    """
    csv_filename = get_ohlcv_file(exchange, symbol, timeframe, start, end)
    csv_filepath = os.path.join(csv_dir, bar_type, csv_filename)
    df = pd.read_csv(
        csv_filepath,
        parse_dates=True,
        date_parser=_parse_datestring,
        header=0,
        sep=',',
        index_col=0,
        # names=['time', 'open', 'high', 'low', 'close', 'volume']
        # names=['time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    df['returns'] = df['close'].pct_change()
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    df.dropna(inplace=True)

    data = df.sort_index()
    return data



def old_open_convert_csv_files(exchange, symbol, timeframe, start, end, csv_dir='data', bar_type='time_bars'):
    """
    Opens the CSV files from the data directory, converting
    them into pandas DataFrames within a symbol dictionary.
    For this handler it will be assumed that the data is
    taken from Yahoo. Thus its format will be respected.
    """
    csv_filename = get_ohlcv_file(exchange, symbol, timeframe, start, end)
    csv_filepath = os.path.join(csv_dir, bar_type, csv_filename)
    df = pd.read_csv(
        csv_filepath,
        parse_dates=True,
        date_parser=_date_parse,
        header=0,
        sep=',',
        index_col=1,
        names=['time', 'open', 'high', 'low', 'close', 'volume']
        # names=['time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    df['returns'] = df['close'].pct_change()
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    df.dropna(inplace=True)

    data = df.sort_index()
    return data

