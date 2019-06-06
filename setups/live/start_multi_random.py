#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    path.append(dir(dir(path[0])))
    __package__ = "live"

import sys
import math
import asyncio
import datetime
import warnings

from itertools import product
from strategies.crypto.multi_random import MultiRandomStrategy
from event import SignalEvent
from trader import CryptoLiveTrade
from datahandler.crypto import LiveDataHandler
from execution.crypto import LiveExecutionHandler
from portfolio import BitmexPortfolio
from datetime import datetime, timedelta
from configuration import Configuration

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    delta = timedelta(minutes=1)
    start_date = datetime.min + math.ceil((datetime.utcnow() - datetime.min) / delta) * delta #round to the next minute

    configuration = Configuration({
      'result_dir' : '../results',
      'exchange_names' : ['bitmex'],
      'assets': { 'bitmex': ['BTC'] },
      'instruments' : { 'bitmex': ['ETH/BTC', 'LTC/BTC', 'TRX/BTC', 'XRP/BTC', 'BTC/USD'] },
      'ohlcv_window': 60, #receive the one minute candles
      'heartbeat' : 1.00,
      'start_date' : start_date,
      'default_position_size': 0.05
    })

    trader = CryptoLiveTrade(
        configuration, LiveDataHandler, LiveExecutionHandler,
        BitmexPortfolio, MultiRandomStrategy
    )

    trader.start_trading()