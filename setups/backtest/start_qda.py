#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(dir(path[0])))
    __package__ = "setups"

import os
import datetime
import sys
import warnings

from strategies.crypto import QDAStrategy
from event import SignalEvent
from trader import CryptoBacktest
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import CryptoPortfolio
from configuration import Configuration

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    configuration = Configuration({
      'csv_dir' :  '../../data',
      'result_dir' : '../../results',
      'file_list' :  ['bitmex-BTCUSD-1d'],
      'exchange_names' :  ['bitmex'],
      'instruments' :  { 'bitmex': ['BTC/USD' ] },
      'ohlcv_window': 86400,
      'initial_capital' :  100000.0,
      'graph_refresh_period': 50,
      'heartbeat' :  0.0,
      'start_date' :  datetime.datetime(2018,1,1),
    })

    backtest = CryptoBacktest(
        configuration, HistoricCSVCryptoDataHandler, SimulatedCryptoExchangeExecutionHandler,
        CryptoPortfolio, QDAStrategy
    )

    backtest.start_trading()
