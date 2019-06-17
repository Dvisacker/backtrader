#!/usr/bin/python
# -*- coding: utf-8 -*-

# mr.py

from __future__ import print_function

from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .strategy import Strategy
from event import SignalEvent, SignalEvents
from trader import CryptoBacktest
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import CryptoPortfolio

class OLSMeanReversionStrategy(Strategy):
    """
    OLS Mean Reversion Strategy v1.1
    Uses ordinary least squares (OLS) to perform a rolling linear
    regression to determine the hedge ratio between a pair of equities.
    The z-score of the residuals time series is then calculated in a
    rolling fashion and if it exceeds an interval of thresholds
    (defaulting to [0.5, 3.0]) then a long/short signal pair are generated
    (for the high threshold) or an exit signal pair are generated (for the
    low threshold).
    """

    def __init__(
        self, data, events, configuration, ols_window=100,
        zscore_exit=1.5, zscore_entry=3
    ):
        """
        Initialises the stat arb strategy.
        Parameters:
        data - The DataHandler object that provides bar information
        events - The Event Queue object.
        """
        self.data = data
        self.instruments = configuration.instruments
        self.events = events
        self.ols_window = ols_window
        self.zscore_exit = zscore_exit
        self.zscore_entry = zscore_entry

        # by default, we simply take the first given exchange
        self.exchanges = configuration.exchange_names
        self.exchange = self.exchanges[0]

        # by default, we simply take the two first given instruments
        self.pair = (self.instruments[self.exchange][0], self.instruments[self.exchange][1])

        self.datetime = datetime.utcnow()
        self.long_market = False
        self.short_market = False

    def calculate_xy_signals(self, zscore_last):
        """
        Calculates the actual x, y signal pairings
        to be sent to the signal generator.
        Parameters
        zscore_last - The current zscore to test against
        """
        y_signal = None
        x_signal = None
        p0 = self.pair[0]
        p1 = self.pair[1]
        dt = self.datetime
        ex = self.exchange
        hr = abs(self.hedge_ratio)

        # If we're long the market and below the
        # negative of the high zscore threshold
        if zscore_last <= -self.zscore_entry and not self.long_market:
            print('LONG,SHORT')
            self.long_market = True
            y_signal = SignalEvent(1, ex, p0, dt, 'LONG', 1.0)
            x_signal = SignalEvent(1, ex, p1, dt, 'SHORT', hr)

        # If we're long the market and between the
        # absolute value of the low zscore threshold
        if abs(zscore_last) <= self.zscore_exit and self.long_market:
            print('EXIT,EXIT')
            self.long_market = False
            y_signal = SignalEvent(1, ex, p0, dt, 'EXIT', 1.0)
            x_signal = SignalEvent(1, ex, p1, dt, 'EXIT', 1.0)

        # If we're short the market and above
        # the high zscore threshold
        if zscore_last >= self.zscore_entry and not self.short_market:
            print('SHORT,LONG')
            self.short_market = True
            y_signal = SignalEvent(1, ex, p0, dt, 'SHORT', 1.0)
            x_signal = SignalEvent(1, ex, p1, dt, 'LONG', hr)

        # If we're short the market and between the
        # absolute value of the low zscore threshold
        if abs(zscore_last) <= self.zscore_exit and self.short_market:
            print('EXIT,EXIT')
            self.short_market = False
            y_signal = SignalEvent(1, ex, p0, dt, 'EXIT', 1.0)
            x_signal = SignalEvent(1, ex, p1, dt, 'EXIT', 1.0)

        return y_signal, x_signal

    def calculate_signals_for_pairs(self):
        """
        Generates a new set of signals based on the mean reversion
        strategy.
        Calculates the hedge ratio between the pair of tickers.
        We use OLS for this, althought we should ideall use CADF.
        """
        # Obtain the latest window of values for each
        # component of the pair of tickers
        y = self.data.get_latest_bars_values(self.exchange, self.pair[0],  "close", N=self.ols_window)
        x = self.data.get_latest_bars_values(self.exchange, self.pair[1], "close", N=self.ols_window)
        signals = []

        if y is not None and x is not None:
            # Check that all window periods are available
            if len(y) >= self.ols_window and len(x) >= self.ols_window and len(x) == len(y):
                # Calculate the current hedge ratio using  OLS
                self.hedge_ratio = sm.OLS(y, x).fit().params[0]

                # Calculate the current z-score of the residuals
                spread = y - self.hedge_ratio * x
                zscore_last = ((spread - spread.mean())/spread.std())[-1]

                # Calculate signals and add to events queue
                y_signal, x_signal = self.calculate_xy_signals(zscore_last)

                if y_signal is not None and x_signal is not None:
                    signals.append(x_signal)
                    signals.append(y_signal)

                    events = SignalEvents(signals, 1)
                    self.events.put(events)


    def calculate_signals(self, event):
        """
        Calculate the SignalEvents based on market data.
        """
        if event.type == 'MARKET':
            self.calculate_signals_for_pairs()