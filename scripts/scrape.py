#!/usr/bin/env python3

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "scripts"

import math
import time
import ccxt
import argparse
import pandas as pd

from datetime import datetime, timedelta, timezone
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation

def parse_args():
    parser = argparse.ArgumentParser(description='Market data downloader')


    parser.add_argument('-s','--symbol',
                        type=str,
                        required=True,
                        help='The Symbol of the Instrument/Currency Pair To Download')

    parser.add_argument('-e','--exchange',
                        type=str,
                        required=True,
                        help='The exchange to download from')

    parser.add_argument('-t','--timeframe',
                        type=str,
                        default='1d',
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
symbol = from_standard_to_exchange_notation(args.exchange, args.symbol, index=True)

# Get our Exchange
try:
    exchange = getattr (ccxt, args.exchange) ()
except AttributeError:
    print('-'*36,' ERROR ','-'*35)
    print('Exchange "{}" not found. Please check the exchange is supported.'.format(args.exchange))
    print('-'*80)
    quit()

# Check if fetching of OHLC Data is supported
if exchange.has["fetchOHLCV"] != True:
    print('-'*36,' ERROR ','-'*35)
    print('{} does not support fetching OHLC data. Please use another exchange'.format(args.exchange))
    print('-'*80)
    quit()

# Check requested timeframe is available. If not return a helpful error.
if (not hasattr(exchange, 'timeframes')) or (args.timeframe not in exchange.timeframes):
    print('-'*36,' ERROR ','-'*35)
    print('The requested timeframe ({}) is not available from {}\n'.format(args.timeframe,args.exchange))
    print('Available timeframes are:')
    for key in exchange.timeframes.keys():
        print('  - ' + key)
    print('-'*80)
    quit()

# Check if the symbol is available on the Exchange
exchange.load_markets()
if symbol not in exchange.symbols:
    print('-'*36,' ERROR ','-'*35)
    print('The requested symbol ({}) is not available from {}\n'.format(symbol,args.exchange))
    print('Available symbols are:')
    for key in exchange.symbols:
        print('  - ' + key)
    print('-'*80)
    quit()


ohlcv = []
# Get OHLCV from now - days to now
if not args.from_date and not args.to_date:
  days = timedelta(days=args.days)
  now = datetime.now()
  current_time = now - days

  timedelta = {
    '1m': timedelta(hours=12),
    '1h': timedelta(days=30),
    '1d': timedelta(days=365)
  }[args.timeframe]

  limit = {
    '1m': 720,
    '1h': 720,
    '1d': 365
  }[args.timeframe]

  while current_time < now:
    print(current_time.timetuple())
    since = time.mktime(current_time.timetuple()) * 1000
    data = exchange.fetch_ohlcv(symbol, args.timeframe, since=since, limit=limit)

    header = ['Time', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    current_time += timedelta
    parser = lambda x : { 'timestamp': x[0] / 1000, 'open': x[1], 'high': x[2], 'low': x[3], 'close': x[4], 'volume': x[5], 'time': datetime.fromtimestamp(x[0] / 1000) }
    parsed_ohlcv = list(map(parser, data))
    ohlcv += parsed_ohlcv

  # Save it
  df = pd.DataFrame(ohlcv, columns=['timestamp', 'time', 'open', 'high', 'low', 'close', 'volume'])
  df.set_index('time')
  symbol_out = symbol.replace("/","")
  filename = '{}-{}-{}.csv'.format(args.exchange, symbol_out,args.timeframe)
  df.to_csv(filename)


# Get OHLCV from from_date to from_date + days
if args.from_date and args.days:
  days = timedelta(days=args.days)
  now = datetime.now()
  current_time = datetime.strptime(args.from_date, '%d/%m/%Y')
  to_date = current_time + days

  timedelta = {
    '1m': timedelta(hours=12),
    '1h': timedelta(days=30),
    '1d': timedelta(days=365)
  }[args.timeframe]

  limit = {
    '1m': 720,
    '1h': 720,
    '1d': 365
  }[args.timeframe]

  while current_time < to_date:
    print(current_time.timetuple())
    since = time.mktime(current_time.timetuple()) * 1000
    data = exchange.fetch_ohlcv(symbol, args.timeframe, since=since, limit=limit)
    header = ['Time', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    current_time += timedelta
    parser = lambda x : { 'timestamp': x[0] / 1000, 'open': x[1], 'high': x[2], 'low': x[3], 'close': x[4], 'volume': x[5], 'time': datetime.fromtimestamp(x[0] / 1000) }
    parsed_ohlcv = list(map(parser, data))
    ohlcv += parsed_ohlcv

# Save it
df = pd.DataFrame(ohlcv, columns=['timestamp', 'time', 'open', 'high', 'low', 'close', 'volume'])
df.set_index('time', inplace=True)
print(df.head())
symbol_out = from_exchange_to_standard_notation(args.exchange, symbol).replace('/', "")
filename = '../data/{}-{}-{}.csv'.format(args.exchange, symbol_out, args.timeframe)
df.to_csv(filename)