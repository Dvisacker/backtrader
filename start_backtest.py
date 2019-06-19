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
from configuration import Configuration, MultiMRConfiguration

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
    "moving_average_crossover": MovingAverageCrossoverStrategy,
    "macd_crossover": MACDCrossover,
    "condition": ConditionBasedStrategy
  }

  backtesters = {
    "crypto_backtest": SimpleBacktest,
    "multi_params_backtest": MultiBacktest,
    "multi_instrument_backtest": MultiInstrumentsBacktest,
    "multi_conditions_backtest": MultiConditionsBacktest,
    "multi_periods_backtest": MultiPeriodsBacktest
  }

  portfolios = {
    "crypto_portfolio": CryptoPortfolio,
    "bitmex_portfolio": BitmexPortfolioBacktest
  }

  if (data['backtester_type'] == "multi_params_backtest"):
    configuration = MultiMRConfiguration(data)
  else:
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