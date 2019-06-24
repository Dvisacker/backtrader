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

class MovingAverageCrossoverStrategy(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy with a
    short/long simple weighted moving average. Default short/long
    windows are 100/400 periods respectively.
    """

    def __init__(self, data, events, configuration, short_window=10, long_window=30):
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
        self.short_window = short_window
        self.long_window = long_window

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
            bought[e][s] = 'OUT'

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
                data = self.data.get_latest_bars_values(e, s, "close", N=self.long_window)

                bar_date = self.data.get_latest_bar_datetime(e, s)
                if data is not None and data != []:

                    # There are two ways to validate whether a signal corresponds to a crossover.
                    # Either, we compute the crossover by comparing signals at successive indexes.
                    # Either, we keep track of whether we are in position or not
                    short_sma = np.mean(data[-self.short_window:])
                    long_sma = np.mean(data[-self.long_window:])

                    dt = datetime.datetime.utcnow()
                    sig_dir = ""

                    if short_sma > long_sma and self.bought[e][s] == 'OUT':
                        print("LONG: {}".format(bar_date))
                        sig_dir = 'LONG'
                        signals = [SignalEvent(1, e, s, dt, sig_dir, 1.0)]
                        signal_events = SignalEvents(signals, 1)
                        self.bought[e][s] = 'LONG'
                        self.events.put(signal_events)
                    elif short_sma < long_sma and self.bought[e][s] == 'LONG':
                        print("EXIT: {}".format(bar_date))
                        sig_dir = 'EXIT'
                        signals = [SignalEvent(1, e, s, dt, sig_dir, 1.0)]
                        signal_events = SignalEvents(signals, 1)
                        self.events.put(signal_events)
                        self.bought[e][s] = 'OUT'