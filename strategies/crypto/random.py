#!/usr/bin/python
# -*- coding: utf-8 -*-

# mr.py

from __future__ import print_function

from datetime import datetime

import random

from .strategy import Strategy
from event import SignalEvent, SignalEvents
from trader import CryptoBacktest
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import CryptoPortfolio

class RandomStrategy(Strategy):
    """
    This strategy simply decides to enter or exit the market randomly
    """

    def __init__(self, data, events, configuration):
        """
        Parameters:
        data - The DataHandler object that provides bar information
        events - The Event Queue object.
        """
        self.data = data
        self.instruments = configuration.instruments
        self.events = events

        # by default, we simply take the first given exchange
        self.exchanges = configuration.exchange_names
        self.exchange = self.exchanges[0]

        # by default, we simply take the two first given instruments
        self.pair = (self.instruments[self.exchange][0], self.instruments[self.exchange][1])

        self.datetime = datetime.utcnow()
        self.long_market = False
        self.short_market = False

    def calculate_signals(self, event):
        """
        Calculate the SignalEvents randomly
        """
        if event.type == 'MARKET':
          y_signal = None
          x_signal = None
          id = None
          p0 = self.pair[0]
          p1 = self.pair[1]
          dt = self.datetime
          ex = self.exchange

          if self.long_market:
            if random.choice([True, False]):
              print('EXIT,EXIT')
              id = 'EXIT,EXIT'
              self.long_market = False
              y_signal = SignalEvent(1, ex, p0, dt, 'EXIT', 1.0)
              x_signal = SignalEvent(1, ex, p1, dt, 'EXIT', 1.0)

          elif self.short_market:
            if random.choice([True, False]):
              print('EXIT,EXIT')
              id = 'EXIT,EXIT'
              self.short_market = False
              y_signal = SignalEvent(1, ex, p0, dt, 'EXIT', 1.0)
              x_signal = SignalEvent(1, ex, p1, dt, 'EXIT', 1.0)

          else:
            choice = random.choice([1,2,3,4])
            if choice == 1:
              print('LONG,SHORT')
              id = 'LONG,SHORT'
              self.long_market = True
              y_signal = SignalEvent(1, ex, p0, dt, 'LONG', 1.0)
              x_signal = SignalEvent(1, ex, p1, dt, 'SHORT', 1.0)
            elif choice == 2:
              print('SHORT,LONG')
              id = 'SHORT,LONG'
              self.short_market = True
              y_signal = SignalEvent(1, ex, p0, dt, 'SHORT', 1.0)
              x_signal = SignalEvent(1, ex, p1, dt, 'LONG', 1.0)

          if y_signal is not None and x_signal is not None:
              events = SignalEvents([x_signal, y_signal], id)
              self.events.put(events)