import sys
import json
import warnings
import argparse

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

  return parser.parse_args()

args = parse_args()

with open(args.file) as f:
  data = json.load(f)

  strategies = {
    "qda": QDAStrategy,
    "rsi": RSIStrategy,
    "random": MultiRandomStrategy,
    "mean_reversion": OLSMeanReversionStrategy,
    "moving_average_crossover": MovingAverageCrossoverStrategy,
    "macd_crossover": MACDCrossover
  }

  backtesters = {
    "crypto_backtest": SimpleBacktest,
    "multi_backtest": MultiBacktest,
    "multi_instrument_backtest": MultiInstrumentsBacktest,
    "multi_conditions_backtest": MultiConditionsBacktest,
    "multi_periods_backtest": MultiPeriodsBacktest
  }

  portfolios = {
    "crypto_portfolio": CryptoPortfolio,
    "bitmex_portfolio": BitmexPortfolioBacktest
  }

  configuration = Configuration(data)
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