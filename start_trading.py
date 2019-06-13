#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import json
import argparse
import warnings

from event import SignalEvent
from trader import CryptoLiveTrade
from strategies.crypto import *
from datahandler.crypto import LiveDataHandler
from execution.crypto import LiveExecutionHandler
from portfolio import BitmexPortfolio
from configuration import Configuration

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def parse_args():
  parser = argparse.ArgumentParser(description='Backtest')
  parser.add_argument('-f', '--file',
                      type=str,
                      require=True,
                      help='The name of the configuration JSON file')

  return parser.parse_args()

args = parse_args()

with open(args.file) as f:
  data = json.load(f)

  strategies = {
    "qda": QDAStrategy,
    "random": MultiRandomStrategy,
    "mean_reversion": OLSMeanReversionStrategy,
    "moving_average_crossover": MovingAverageCrossoverStrategy
  }

  configuration = Configuration(data)
  strategy_class = strategies[data['strategy']]

  trader = CryptoLiveTrade(
    configuration,
    LiveDataHandler,
    LiveExecutionHandler,
    BitmexPortfolio,
    strategy_class
  )

  trader.start_trading()

