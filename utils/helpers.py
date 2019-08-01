import time, urllib, hmac, hashlib, math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime

from indicators import *


def truncate(n, decimals=0):
  multiplier = 10 ** decimals
  return int(n * multiplier) / multiplier

def convert_order_type(order_type):
  return {
    "StopLoss": "Stop",
    "TakeProfit": "MarketIfTouched"
  }[order_type]

def get_precision(symbol):
  return {
     "BCH/BTC": 4,
      "BCHM19": 4,
     "EOS/BTC": 7,
      "EOSM19": 7,
     "ETH/BTC": 4,
      "ETHM19": 4,
      "ETHXBT": 4,
     "LTC/BTC": 5,
      "LTCM19": 5,
     "TRX/BTC": 7,
      "TRXM19": 7,
     "ADA/BTC": 7,
      "ADAM19": 7,
     "XRP/BTC": 8,
      "XRPM19": 8,
     "BTC/USD": 0,
      "XBTM19": 0,
     "ETH/USD": 1,
      "ETHUSD": 1,
  }[symbol]

def date_parse(timestamp):
    """
    Parses timestamps into python datetime objects.
    """
    return datetime.fromtimestamp(float(timestamp))

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def plot():
  try:
    plt.show()
  except UnicodeDecodeError:
    plot()


def create_lagged_crypto_series(data, start_date, end_date, lags=5):
    """
    This creates a pandas DataFrame that stores the
    percentage returns of the adjusted closing value of
    a stock obtained from Yahoo Finance, along with a
    number of lagged returns from the prior trading days
    (lags defaults to 5 days). Trading volume, as well as
    the Direction from the previous day, are also included.
    """

    data_lagged = pd.DataFrame(index=data.index)
    data_lagged["Today"] = data['close']
    data_lagged["Volume"] = data['volume']

    for i in range(0, lags):
        data_lagged["Lag%s" % str(i+1)] = data['close'].shift(i+1)

    # Create the returns DataFrame
    data_returns = pd.DataFrame(index=data_lagged.index)
    data_returns["Volume"] = data_lagged["Volume"]
    data_returns["Today"] = data_lagged["Today"].pct_change()*100.0

    for i, x in enumerate(data_returns["Today"]):
        if (abs(x) < 0.0001):
            data_returns["Today"][i] = 0.0001

    for i in range(0, lags):
        data_returns[
            "Lag%s" % str(i+1)
        ] = data_lagged["Lag%s" % str(i+1)].pct_change()*100.0


    data_returns["Direction"] = np.sign(data_returns["Today"])
    data_returns = data_returns[data_returns.index >= start_date]

    return data_returns

def generate_nonce():
    return int(round(time.time() + 3600))


# Generates an API signature.
# A signature is HMAC_SHA256(secret, verb + path + nonce + data), hex encoded.
# Verb must be uppercased, url is relative, nonce must be an increasing 64-bit integer
# and the data, if present, must be JSON without whitespace between keys.
#
# For example, in psuedocode (and in real code below):
#
# verb=POST
# url=/api/v1/order
# nonce=1416993995705
# data={"symbol":"XBTZ14","quantity":1,"price":395.01}
# signature = HEX(HMAC_SHA256(secret, 'POST/api/v1/order1416993995705{"symbol":"XBTZ14","quantity":1,"price":395.01}'))
def generate_signature(secret, verb, url, nonce, data):
    """Generate a request signature compatible with BitMEX."""
    # Parse the url so we can remove the base and extract just the path.
    parsedURL = urllib.parse.urlparse(url)
    path = parsedURL.path
    if parsedURL.query:
        path = path + '?' + parsedURL.query

    # print "Computing HMAC: %s" % verb + path + str(nonce) + data
    message = (verb + path + str(nonce) + data).encode('utf-8')

    signature = hmac.new(secret.encode('utf-8'), message, digestmod=hashlib.sha256).hexdigest()
    return signature

def ceil_dt(dt, delta):
    return datetime.min + math.ceil((dt - datetime.min) / delta) * delta

def to_standard_notation(symbol):
  symbol = symbol.replace("XBT", "BTC")
  return symbol

def to_bitmex_notation(symbol):
  symbol = symbol.replace("BTC", "XBT")
  symbol = symbol.replace("/", "")
  return symbol

def from_bitmex_notation(symbol):
  symbol = symbol.replace("XBTUSD", "BTC/USD")
  symbol = symbol.replace("ETHUSD", "ETH/USD")
  return symbol

def from_file_to_standard_notation(symbol):
  if symbol == 'BTCUSD':
    return 'BTC/USD'
  elif symbol == 'ETHUSD':
    return 'ETH/USD'
  else:
    return symbol

def from_standard_to_file_notation(symbol):
  symbol = symbol.replace('/', "")
  return symbol

def from_exchange_to_standard_notation(exchange, symbol):
  if exchange == 'bitmex':
    if symbol == 'XBTUSD':
      return 'BTC/USD'
    elif symbol == 'ETHUSD':
      return 'ETH/USD'
    elif symbol == 'ADAU19':
      return 'ADA/BTC'
    elif symbol == "BCHU19":
      return 'BCH/BTC'
    elif symbol == "EOSU19":
      return 'EOS/BTC'
    elif symbol == "ETHU19":
      return 'ETH/BTC'
    elif symbol == "LTCU19":
      return 'LTC/BTC'
    elif symbol == "TRXU19":
      return 'TRX/BTC'
    elif symbol == "XRPU19":
      return 'XRP/BTC'
    elif symbol == "XBTU19":
      return 'BTC/USD'
    elif symbol == '.BADAXBT':
      return 'ADA/BTC'
    elif symbol == ".BBCHXBT":
      return 'BCH/BTC'
    elif symbol == ".BEOSXBT":
      return 'EOS/BTC'
    elif symbol == ".BETHXBT":
      return 'ETH/BTC'
    elif symbol == ".BLTCXBT":
      return 'LTC/BTC'
    elif symbol == ".TRXXBT":
      return 'TRX/BTC'
    elif symbol == ".BXRPXBT":
      return 'XRP/BTC'
    elif symbol == ".BXBTXBT":
      return 'BTC/USD'
    else:
      return symbol
  else:
    return symbol

def from_standard_to_exchange_notation(exchange, symbol, index=False):
  if exchange == 'bitmex' and index == False:
    if symbol == "ADA/BTC":
      return "ADAU19"
    elif symbol == "BCH/BTC":
      return "BCHU19"
    elif symbol == "EOS/BTC":
      return "EOSU19"
    elif symbol == "ETH/BTC":
      return "ETHU19"
    elif symbol == "LTC/BTC":
      return "LTCU19"
    elif symbol == "TRX/BTC":
      return "TRXU19"
    elif symbol == "XRP/BTC":
      return "XRPU19"
    elif symbol == "BTC/USD":
      return "XBTU19"
    elif symbol == "ETH/USD":
      return "ETH/USD"

  if exchange == 'bitmex' and index == True:
    if symbol == "ADA/BTC":
      return ".BADAXBT"
    elif symbol == "BCH/BTC":
      return ".BBCHXBT"
    elif symbol == "EOS/BTC":
      return ".BEOSXBT"
    elif symbol == "ETH/BTC":
      return ".BETHXBT"
    elif symbol == "LTC/BTC":
      return ".BLTCXBT"
    elif symbol == "TRX/BTC":
      return ".TRXXBT"
    elif symbol == "XRP/BTC":
      return ".BXRPXBT"
    elif symbol == "BTC/USD":
      return "BTC/USD"
    elif symbol == "ETH/USD":
      return "ETH/USD"

    symbol = symbol.replace('BTC', 'XBT')
    symbol = symbol.replace('/', "")
    return symbol
  else:
    return symbol

def get_timeframe(period):
  if period == 60:
    return '1m'
  elif period == 300:
    return '5m'
  elif period == 900:
    return '15m'
  elif period == 3600:
    return '1h'
  elif period == 86400:
    return '1d'
  else:
    raise Exception('OHLCV window invalid')

def get_data_file(exchange, symbol, period):
  symbol = from_standard_to_file_notation(symbol)
  return '{}-{}-{}.csv'.format(exchange, symbol, period)


def get_bars_file(exchange, symbol, timeframe, start_date, end_date):
  symbol = from_standard_to_file_notation(symbol)



def get_ohlcv_file(exchange, symbol, period, start_date, end_date):
  symbol = from_standard_to_file_notation(symbol)
  start_date = start_date.replace("/", "")
  end_date = end_date.replace("/", "")
  return '{}-{}-{}-{}-{}.csv'.format(exchange, symbol, period, start_date, end_date)




def to_ccxt_notation(symbol):
  symbol = symbol.replace("")

def compute_indicators(prices, indicators):
  df = pd.DataFrame()
  df['prices'] = prices
  for indicator in indicators:
    name = indicator['name']
    params = indicator['params']

    function_map = {
      "rsi": rsi,
      "mavg": mavg,
      "macd": macd_trigger_line,
    }

    function = function_map[name]
    df[name] = function(prices, **params)

  return df


def compute_indicator(prices, indicator):
  df = pd.DataFrame()
  df['prices'] = prices

  name = indicator['name']
  params = indicator['params']

  function_map = {
    "rsi": rsi,
    "mavg": mavg,
    "macd": macd_trigger_line,
  }

  function = function_map[name]
  return function(prices, **params)

def compute_all_indicators(instruments, data, indicators):
  df = pd.DataFrame()

  for s in instruments:
    for i in indicators:
      p = data.get_all_bars_values('bitmex', s, 'close')
      indicator_values = compute_indicator(p, i)
      df['bitmex-{}-{}'.format(s,i)] = indicator_values

  return df


def merge(dict1, dict2):
  res = {**dict1, **dict2}
  return res


def format_instrument_list(instruments):
  return '|'.join(x + ':' + '.'.join(y for y in instruments[x]) for x in instruments)
