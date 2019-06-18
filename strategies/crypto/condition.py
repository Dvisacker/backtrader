import datetime
import talib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .strategy import Strategy
from event import SignalEvent, SignalEvents
from trader import SimpleBacktest
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import CryptoPortfolio

class ConditionBasedStrategy(Strategy):
    """
    Carries out a strategy based on long, short and exit positions given as parameters
    """

    def __init__(self, data, events, configuration, conditions):
        """
        Initialises the Moving Average Cross Strategy.
        :param data: The DataHandler object that provides bar information.
        :param events: The Event Queue object.
        :param short_window: The short moving average lookback.
        :param long_window: The long moving average lookback.
        """
        self.data = data
        self.instruments = configuration.instruments
        self.exchanges = configuration.exchange_names
        self.exchange = self.exchanges[0]
        self.events = events
        self.long_condition = conditions['long']
        self.exit_condition = conditions['exit']
        self.short_condition = conditions['short']

        # Set to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to ’OUT’.
        """
        bought = dict( (k,v) for k,v in [(e, {}) for e in self.instruments])
        e = self.exchange
        for s in self.instruments[e]:
            bought[e][s] = 'EXIT'

        return bought

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
                prices = self.data.get_latest_bars_values(e, s, "close", N=100)

                bar_date = self.data.get_latest_bar_datetime(e, s)
                if prices is not None and prices != []:
                    # There are two ways to validate whether a signal corresponds to a crossover.
                    # Either, we compute the crossover by comparing signals at successive indexes.
                    # Either, we keep track of whether we are in position or not
                    dt = datetime.datetime.utcnow()
                    sig_dir = ""

                    if self.long_condition(prices) and self.bought[e][s] != 'LONG':
                        print("LONG: {}".format(bar_date))
                        sig_dir = 'LONG'
                        signals = [SignalEvent(1, e, s, dt, sig_dir, 1.0)]
                        signal_events = SignalEvents(signals, 1)
                        self.bought[e][s] = 'LONG'
                        self.events.put(signal_events)
                    elif self.short_condition(prices) and self.bought[e][s] != 'SHORT':
                        print("SHORT: {}".format(bar_date))
                        sig_dir = 'SHORT'
                        signals = [SignalEvent(1, e, s, dt, sig_dir, 1.0)]
                        signal_events = SignalEvents(signals, 1)
                        self.events.put(signal_events)
                        self.bought[e][s] = 'SHORT'
                    elif self.exit_condition(prices) and self.bought[e][s] != 'EXIT':
                        print("EXIT: {}".format(bar_date))
                        sig_dir = 'EXIT'
                        signals = [SignalEvent(1, e, s, dt, sig_dir, 1.0)]
                        signal_events = SignalEvents(signals, 1)
                        self.events.put(signal_events)
                        self.bought[e][s] = 'EXIT'
