import time, urllib, hmac, hashlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime

def truncate(n, decimals=0):
  multiplier = 10 ** decimals
  return int(n * multiplier) / multiplier

def date_parse(timestamp):
    """
    Parses timestamps into python datetime objects.
    """

    return datetime.fromtimestamp(int(timestamp))

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


def create_lagged_series(symbol, start_date, end_date, lags=5):
    """
    This creates a pandas DataFrame that stores the
    percentage returns of the adjusted closing value of
    a stock obtained from Yahoo Finance, along with a
    number of lagged returns from the prior trading days
    (lags defaults to 5 days). Trading volume, as well as
    the Direction from the previous day, are also included.
    """

    # Obtain stock information from Yahoo Finance
    ts = DataReader(
    	symbol, "yahoo",
    	start_date-datetime.timedelta(days=365),
    	end_date
    )

    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]

    # Create the shifted lag series of prior trading period close values
    for i in range(0,lags):
        tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0,lags):
        tsret[
            "Lag%s" % str(i+1)
        ] = tslag["Lag%s" % str(i+1)].pct_change()*100.0

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]

    return tsret


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
  if symbol == 'BTC/USD':
    return 'BTCUSD'
  elif symbol == 'ETH/USD':
    return 'ETHUSD'
  else:
    return symbol

def from_exchange_to_standard_notation(exchange, symbol):
  if exchange == 'bitmex':
    if symbol == 'XBTUSD':
      return 'BTC/USD'
    elif symbol == 'ETHUSD':
      return 'ETH/USD'
    elif symbol == 'ADAM19':
      return 'ADA/BTC'
    elif symbol == "BCHM19":
      return 'BCH/BTC'
    elif symbol == "EOSM19":
      return 'EOS/BTC'
    elif symbol == "ETHM19":
      return 'ETH/BTC'
    elif symbol == "LTCM19":
      return 'LTC/BTC'
    elif symbol == "TRXM19":
      return 'TRX/BTC'
    elif symbol == "XRPM19":
      return 'XRP/BTC'
    elif symbol == "XBTM19":
      return 'BTC/USD'
    else:
      return symbol
  else:
    return symbol

def from_standard_to_exchange_notation(exchange, symbol):
  if exchange == 'bitmex':
    if symbol == "ADA/BTC":
      return "ADAM19"
    elif symbol == "BCH/BTC":
      return "BCHM19"
    elif symbol == "EOS/BTC":
      return "EOSM19"
    elif symbol == "ETH/BTC":
      return "ETHM19"
    elif symbol == "LTC/BTC":
      return "LTCM19"
    elif symbol == "TRX/BTC":
      return "TRXM19"
    elif symbol == "XRP/BTC":
      return "XRPM19"
    elif symbol == "BTC/USD":
      return "XBTM19"
    elif symbol == "ETH/USD":
      return "ETHUSD"

    symbol = symbol.replace('BTC', 'XBT')
    symbol = symbol.replace('/', "")
    return symbol
  else:
    return symbol


def get_ohlcv_window(period):
  if period == 60:
    return '1m'
  elif period == 300:
    return '5m'
  elif period == 3600:
    return '1h'
  elif period == 86400:
    return '1d'
  else:
    raise Exception('OHLCV window invalid')

def get_data_file(exchange, symbol, period):
  symbol = from_standard_to_file_notation(symbol)
  return '{}-{}-{}.csv'.format(exchange, symbol, period)

def to_ccxt_notation(symbol):
  symbol = symbol.replace("")