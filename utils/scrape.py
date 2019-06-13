import ccxt
import time
import pandas as pd

from datetime import timedelta, datetime
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation

def scrape_ohlcv(exchange, symbol, timeframe, from_datestring, to_datestring):
    symbol = from_standard_to_exchange_notation(exchange, symbol, index=True)

    # Get our Exchange
    try:
        exchange_instance = getattr (ccxt, exchange) ()
    except AttributeError:
        print('-'*36,' ERROR ','-'*35)
        print('Exchange "{}" not found. Please check the exchange is supported.'.format(exchange))
        print('-'*80)
        quit()

    # Check if fetching of OHLC Data is supported
    if exchange_instance.has["fetchOHLCV"] != True:
        print('-'*36,' ERROR ','-'*35)
        print('{} does not support fetching OHLC data. Please use another exchange'.format(exchange))
        print('-'*80)
        quit()

    # Check requested timeframe is available. If not return a helpful error.
    if (not hasattr(exchange_instance, 'timeframes')) or (timeframe not in exchange_instance.timeframes):
        print('-'*36,' ERROR ','-'*35)
        print('The requested timeframe ({}) is not available from {}\n'.format(timeframe,exchange))
        print('Available timeframes are:')
        for key in exchange.timeframes.keys():
            print('  - ' + key)
        print('-'*80)
        quit()

    # Check if the symbol is available on the Exchange
    exchange_instance.load_markets()
    if symbol not in exchange_instance.symbols:
        print('-'*36,' ERROR ','-'*35)
        print('The requested symbol ({}) is not available from {}\n'.format(symbol,exchange))
        print('Available symbols are:')
        for key in exchange_instance.symbols:
            print('  - ' + key)
        print('-'*80)
        quit()

    from_date = datetime.strptime(from_datestring, '%d/%m/%Y')
    current_time = from_date
    to_date = datetime.strptime(to_datestring, '%d/%m/%Y')

    delta = {
      '1m': timedelta(hours=12),
      '1h': timedelta(days=30),
      '1d': timedelta(days=365)
    }[timeframe]

    limit = {
      '1m': 720,
      '1h': 720,
      '1d': 365
    }[timeframe]

    ohlcv = []
    while current_time < to_date:
      while True:
        try:
          since = time.mktime(current_time.timetuple()) * 1000
          data = exchange_instance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
          current_time += delta
          parser = lambda x : { 'timestamp': x[0] / 1000, 'open': x[1], 'high': x[2], 'low': x[3], 'close': x[4], 'volume': x[5], 'time': datetime.fromtimestamp(x[0] / 1000) }
          parsed_ohlcv = list(map(parser, data))
          ohlcv += parsed_ohlcv
        except Exception as e:
          if e.__class__.__name__ == "DDoSProtection":
            print('Download is being rate-limited. Retrying in 2 seconds')
            time.sleep(2)
            continue
          else:
            print('Error:', e)
            break

        break


    df = pd.DataFrame(ohlcv, columns=['timestamp', 'time', 'open', 'high', 'low', 'close', 'volume'])
    df.set_index('time', inplace=True)

    symbol_out = from_exchange_to_standard_notation(exchange, symbol).replace('/', "")
    from_datestring = from_datestring.replace("/", "")
    to_datestring = to_datestring.replace("/", "")
    filename = 'data/{}-{}-{}-{}-{}.csv'.format(exchange, symbol_out, timeframe, from_datestring, to_datestring)

    df.to_csv(filename)