import talib
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .strategy import Strategy
from event import SignalEvent, SignalEvents
from trader import SimpleBacktest
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import CryptoPortfolio
from utils.log import logger

from indicators import mean_momentum

class OnlyShortMomentumStrategy(Strategy):
    """
    Carries out a basic Momentum Strategy
    """

    def __init__(self, data, events, configuration, short_period=100, long_period=100, zscore_exit=1, zscore_entry=4):
        """
        Initialises the Moving Average Cross Strategy.
        :param data: The DataHandler object that provides bar information.
        :param events: The Event Queue object.
        :param short_window: The short moving average lookback.
        :param long_window: The long moving average lookback.
        """
        self.strategy_name = "only_short_momentum"
        self.data = data
        self.instruments = configuration.instruments
        self.exchanges = configuration.exchange_names
        self.exchange = self.exchanges[0]
        self.events = events
        self.zscore_exit = zscore_exit
        self.zscore_entry = zscore_entry
        self.short_period = short_period
        self.long_period = long_period

        # Set to True if a symbol is in the market
        self.market_status = self._calculate_initial_market_status()

    def _calculate_initial_market_status(self):
        """
        Adds keys to the market_status dictionary for all symbols
        and sets them to ’OUT’.
        """
        market_status = dict( (k,v) for k,v in [(e, {}) for e in self.instruments])
        e = self.exchange
        for s in self.instruments[e]:
            market_status[e][s] = 'EXIT'

        return market_status

    def calculate_signals(self, event):
        """
        Generates a new set of signals based on the MAC
        SMA with the short window crossing the long window
        meaning a long entry and vice versa for a short entry.
        :param event: event - A MarketEvent object.
        """
        e = self.exchange

        if event.type == 'MARKET':
            for s in self.instruments[e]:
                prices = self.data.get_latest_bars_values(e, s, "close", N=self.long_period)

                if prices is not None:
                  bar_date = self.data.get_latest_bar_datetime(e, s)
                  # long_period_returns = (prices[-self.short_period] - prices[-self.long_period]) / (prices[-self.long_period])
                  # short_period_returns = (prices[-1] - prices[-self.short_period]) / (prices[-self.short_period])
                  # momentum = (short_period_returns - long_period_returns) / long_period_returns
                  momentum = mean_momentum(prices, self.short_period, self.long_period)
                  dt = datetime.datetime.utcnow()
                  if momentum <= -self.zscore_entry and self.market_status[e][s] != 'SHORT':
                      logger.info("SHORT: {}".format(bar_date))
                      sig_dir = 'SHORT'
                      signals = [SignalEvent(1, e, s, dt, sig_dir, 1.0)]
                      signal_events = SignalEvents(signals, 1)
                      self.events.put(signal_events)
                      self.market_status[e][s] = 'SHORT'
                  elif abs(momentum) <= self.zscore_exit and self.market_status[e][s] != 'EXIT':
                      logger.info("EXIT: {}".format(bar_date))
                      sig_dir = 'EXIT'
                      signals = [SignalEvent(1, e, s, dt, sig_dir, 1.0)]
                      signal_events = SignalEvents(signals, 1)
                      self.events.put(signal_events)
                      self.market_status[e][s] = 'EXIT'




