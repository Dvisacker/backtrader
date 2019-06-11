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
from portfolio import BitmexPortfolioBacktest
from configuration import Configuration

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    configuration = Configuration({
      'start_date' :  datetime.datetime(2018,1,1),
      'csv_dir' :  '../../data',
      'result_dir' : '../../results',
      'file_list' :  ['bitmex-BTCUSD-1d'],
      'feeds': { 'bitmex': ['XRP/BTC', 'BTC/USD']},
      'instruments': { 'bitmex': ['XRP/BTC']},
      'assets': { 'bitmex': ['BTC']},
      'exchange_names' :  ['bitmex'],
      'ohlcv_window': '1d',
      'initial_capital' :  100000.0,
      'graph_refresh_period': 50,
      'heartbeat' :  0.0,
      'default_position_size': 0.05,
      'update_charts': False,
      'show_charts': True,
      'initial_bars': 30
    })

    backtest = CryptoBacktest(
        configuration, HistoricCSVCryptoDataHandler, SimulatedCryptoExchangeExecutionHandler,
        BitmexPortfolioBacktest, QDAStrategy
    )

    backtest.start_trading()
