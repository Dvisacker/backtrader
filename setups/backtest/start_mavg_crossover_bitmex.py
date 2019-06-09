#!/usr/bin/python
# -*- coding: utf-8 -*-

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(dir(path[0])))
    __package__ = "setups"


import sys
import datetime
import warnings

from strategies.crypto import MovingAverageCrossoverStrategy
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
      'csv_dir': '../../data',
      'result_dir': '../../results',
      'feeds': { 'bitmex': ['EOS/BTC', 'BTC/USD']},
      'instruments' :  { 'bitmex': ['EOS/BTC'] },
      'assets': { 'bitmex': [ 'BTC' ]},
      'ohlcv_window': 60,
      'initial_capital': 100000.0,
      'heartbeat': 0,
      'start_date': datetime.datetime(2017, 1, 1, 0, 0, 0),
      'graph_refresh_period': 50,
      'default_position_size': 0.05
    })

    backtest = CryptoBacktest(
        configuration,
        HistoricCSVCryptoDataHandler,
        SimulatedCryptoExchangeExecutionHandler,
        BitmexPortfolioBacktest,
        MovingAverageCrossoverStrategy
    )

    backtest.start_trading()