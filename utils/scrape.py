import ccxt
import time
import logging
import requests
import pandas as pd

from datetime import timedelta, datetime
from tqdm import tqdm
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation

def hello():
  return "hi"

def scrape_ohlcv(exchange, symbol, timeframe, from_datestring, to_datestring):
    symbol = from_standard_to_exchange_notation(exchange, symbol, index=True)

    # Get our Exchange
    try:
        exchange_instance = getattr (ccxt, exchange) ()
    except AttributeError:
        logging.error('Exchange "{}" not found. Please check the exchange is supported.'.format(exchange))
        logging.error('-'*80)
        quit()

    # Check if fetching of OHLC Data is supported
    if exchange_instance.has["fetchOHLCV"] != True:
        logging.error('{} does not support fetching OHLC data. Please use another exchange'.format(exchange))
        logging.error('-'*80)
        quit()

    # Check if the symbol is available on the Exchange
    exchange_instance.load_markets()
    if symbol not in exchange_instance.symbols:
        logging.error('The requested symbol ({}) is not available from {}\n'.format(symbol,exchange))
        logging.error('Available symbols are:')
        for key in exchange_instance.symbols:
            logging.error('  - ' + key)
        logging.error('-'*80)
        quit()

    from_date = datetime.strptime(from_datestring, '%d/%m/%Y')
    current_time = from_date
    to_date = datetime.strptime(to_datestring, '%d/%m/%Y')

    # Some timeframes are not available. Therefore we download a lower timeframe
    # and downsample the data afterwards.
    download_timeframe = {
      '1m': '1m',
      '5m': '5m',
      '15m': '5m',
      '1h': '1h',
      '1d': '1d'
    }[timeframe]

    delta = {
      '1m': timedelta(hours=12),
      '5m': timedelta(days=2, hours=12),
      '15m': timedelta(days=7, hours=12),
      '1h': timedelta(days=30),
      '1d': timedelta(days=365)
    }[download_timeframe]

    limit = {
      '1m': 720,
      '5m': 720,
      '15m': 720,
      '1h': 720,
      '1d': 365
    }[download_timeframe]



    ohlcv = []
    while current_time < to_date:
      while True:
        try:
          since = time.mktime(current_time.timetuple()) * 1000
          data = exchange_instance.fetch_ohlcv(symbol, download_timeframe, since=since, limit=limit)
          current_time += delta
          parser = lambda x : { 'timestamp': x[0] / 1000, 'open': x[1], 'high': x[2], 'low': x[3], 'close': x[4], 'volume': x[5], 'time': datetime.fromtimestamp(x[0] / 1000) }
          parsed_ohlcv = list(map(parser, data))
          ohlcv += parsed_ohlcv
        except Exception as e:
          if e.__class__.__name__ == "DDoSProtection":
            logging.warning('Download is being rate-limited. Retrying in 2 seconds')
            time.sleep(2)
            continue
          else:
            logging.error(e)
            break

        break

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'time', 'open', 'high', 'low', 'close', 'volume'])
    df.set_index('time', inplace=True)

    # Downsample the timeframe to 15min by dropping every 2 of 3 rows on a 5min dataframe
    if download_timeframe != timeframe:
      if timeframe == '15m':
        df = df.iloc[::3]

    symbol_out = from_exchange_to_standard_notation(exchange, symbol).replace('/', "")
    from_datestring = from_datestring.replace("/", "")
    to_datestring = to_datestring.replace("/", "")
    filename = 'data/{}-{}-{}-{}-{}.csv'.format(exchange, symbol_out, timeframe, from_datestring, to_datestring)

    df.to_csv(filename)


def scrape_bitmex_trades(from_datestring, to_datestring):
    url = 'https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/'
    current_datetime = datetime.strptime(from_datestring, '%d/%m/%Y')
    to_datetime = datetime.strptime(to_datestring, '%d/%m/%Y')

    while current_datetime < to_datetime:
      date_string = current_datetime.strftime('%Y%m%d')
      file_name = date_string + '.csv.gz'
      file_url = url + file_name
      print('Downloading {}'.format(file_name))
      response = requests.get(file_url, stream=True)

      with open(file_name, "wb") as handle:
        for data in response.iter_content():
          handle.write(data)

      print('Downloaded trade data for {}'.format(current_datetime))
      current_datetime += timedelta(days=1)


def scrape_bitmex_quotes(from_datestring, to_datestring):
    url = 'https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/quote/'
    current_datetime = datetime.strptime(from_datestring, '%d/%m/%Y')
    to_datetime = datetime.strptime(to_datestring, '%d/%m/%Y')

    while current_datetime < to_datetime:
      date_string = current_datetime.strftime('%Y%m%d')
      file_name = date_string + '.csv.gz'
      file_url = url + file_name
      print('Downloading {}'.format(file_name))
      response = requests.get(file_url, stream=True)

      with open(file_name, "wb") as handle:
        for data in response.iter_content():
          handle.write(data)

      print('Downloaded trade data for {}'.format(current_datetime))
      current_datetime += timedelta(days=1)

def scrape_trades(exchange, symbol, from_datestring, to_datestring):
    symbol = from_standard_to_exchange_notation(exchange, symbol, index=True)
    # Get our Exchange
    try:
        exchange_instance = getattr (ccxt, exchange) ()
    except AttributeError:
        logging.error('Exchange "{}" not found. Please check the exchange is supported.'.format(exchange))
        logging.error('-'*80)
        quit()

    # Check if fetching of OHLC Data is supported
    if exchange_instance.has["fetchTrades"] != True:
        logging.error('{} does not support fetching OHLC data. Please use another exchange'.format(exchange))
        logging.error('-'*80)
        quit()

    # Check if the symbol is available on the Exchange
    exchange_instance.load_markets()
    if symbol not in exchange_instance.symbols:
        logging.error('The requested symbol ({}) is not available from {}\n'.format(symbol,exchange))
        logging.error('Available symbols are:')
        for key in exchange_instance.symbols:
            logging.error('  - ' + key)
        logging.error('-'*80)
        quit()


    from_date = datetime.strptime(from_datestring, '%d/%m/%Y')
    to_date = datetime.strptime(to_datestring, '%d/%m/%Y')
    current_time = from_date


    limit = 500
    trades = []
    while current_time < to_date:
      while True:
        try:
          since = time.mktime(current_time.timetuple()) * 1000
          data = exchange_instance.fetch_trades(symbol, since=since, limit=limit)
          parser = lambda x : { 'timestamp': x['timestamp'] / 1000, 'price': x['price'], 'side': x['side'], 'amount': x['amount'], 'time': datetime.fromtimestamp(x['timestamp'] / 1000) }
          parsed_trades = list(map(parser, data))
          last_trade = parsed_trades[-1]
          current_time = last_trade['time']
          trades += parsed_trades
        except Exception as e:
          if e.__class__.__name__ == "DDoSProtection":
            logging.warning('Download is being rate-limited. Retrying in 2 seconds')
            time.sleep(2)
            continue
          else:
            logging.error(e)
            break

        break

    df = pd.DataFrame(trades, columns=['timestamp', 'time', 'price', 'amount', 'side'])
    df.set_index('time', inplace=True)

    symbol_out = from_exchange_to_standard_notation(exchange, symbol).replace('/', "")
    from_datestring = from_datestring.replace("/", "")
    to_datestring = to_datestring.replace("/", "")
    filename = 'data/trades-{}-{}-{}-{}.csv'.format(exchange, symbol_out, from_datestring, to_datestring)

    df.to_csv(filename)