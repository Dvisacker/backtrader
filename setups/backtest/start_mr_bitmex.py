#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(dir(path[0])))
    __package__ = "setups"

import sys
import math
import warnings

from strategies.crypto.mr import OLSMeanReversionStrategy
from trader import CryptoBacktest
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import BitmexPortfolioBacktest
from datetime import datetime, timedelta
from configuration import Configuration

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    configuration = Configuration({
      'start_date' : datetime(2019, 3, 26, 0, 0, 0),
      'csv_dir' : '../../data',
      'result_dir' : '../../results',
      'feeds': { 'bitmex': ['XRP/BTC', 'EOS/BTC', 'BTC/USD'] },
      'instruments': { 'bitmex': ['XRP/BTC', 'EOS/BTC'] },
      'assets': { 'bitmex': [ 'BTC' ]},
      'initial_capital' : 100000.0,
      'ohlcv_window': 60,
      'graph_refresh_period': 500,
      'heartbeat' : 0,
      'graph_refresh_period': 300,
      'default_position_size': 0.05,
      'update_charts': False,
      'show_charts': True
    })

    backtest = CryptoBacktest(
        configuration, HistoricCSVCryptoDataHandler, SimulatedCryptoExchangeExecutionHandler,
        BitmexPortfolioBacktest, OLSMeanReversionStrategy
    )

    backtest.start_trading()