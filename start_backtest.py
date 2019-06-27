import sys
import json
import warnings
import argparse
import importlib

from trader import *
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import BitmexPortfolioBacktest, CryptoPortfolio
from strategies.crypto.multi_random import MultiRandomStrategy
from configuration import Configuration

from strategies.crypto import *

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def parse_args():
  parser = argparse.ArgumentParser(description='Backtest')

  parser.add_argument('-f', '--file',
                      type=str,
                      required=True,
                      help='The name of the configuration JSON file')

  parser.add_argument('-c', '--conditions',
                      type=str,
                      required=False,
                      help='The name of the python module containing list of conditions to be tested')

  return parser.parse_args()

args = parse_args()

if args.conditions:
  condition_file = importlib.import_module(args.conditions)

with open(args.file) as f:
  data = json.load(f)
  strategies = {
    "qda": QDAStrategy,
    "rsi": RSIStrategy,
    "random": MultiRandomStrategy,
    "mean_reversion": OLSMeanReversionStrategy,
    "generalized_mean_reversion": GeneralizedMeanReversion,
    "moving_average_crossover": MovingAverageCrossoverStrategy,
    "macd_crossover": MACDCrossover,
    "condition": ConditionBasedStrategy,
    "momentum": MomentumStrategy,
    "only_short_momentum": OnlyShortMomentumStrategy,
    "only_long_momentum": OnlyLongMomentumStrategy
  }

  backtesters = {
    "simple_backtest": SimpleBacktest,
    "multi_params_backtest": MultiParamsBacktest,
    "multi_conditions_backtest": MultiConditionsBacktest,
    "super_backtest": SuperBacktest
  }

  portfolios = {
    "crypto_portfolio": CryptoPortfolio,
    "bitmex_portfolio": BitmexPortfolioBacktest
  }

  configuration = Configuration(data)

  if args.conditions:
    configuration.conditions = condition_file.conditions

  backtester_class = backtesters[data['backtester_type']]
  portfolio_class = portfolios[data['portfolio_type']]
  strategy_class = strategies[data['strategy']]

  backtest = backtester_class(
      configuration,
      HistoricCSVCryptoDataHandler,
      SimulatedCryptoExchangeExecutionHandler,
      portfolio_class,
      strategy_class
  )

  backtest.start_trading()