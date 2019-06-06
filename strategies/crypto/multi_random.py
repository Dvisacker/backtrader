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

class MultiRandomStrategy(Strategy):
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

        self.instruments = self.instruments[self.exchange]

        self.datetime = datetime.utcnow()
        self.state = dict( (k,v) for k, v in [(s, '') for s in self.instruments])

    def calculate_signals(self, event):
        """
        Calculate the SignalEvents randomly
        """
        if event.type == 'MARKET':
          id = 1
          dt = self.datetime
          ex = self.exchange
          signals = []
          print('In strategy')

          for s in self.instruments:
            # For each instrument, we have 1/10 chance to take a position. Respectively 1/20 and 1/20 for long/short
            choice = random.choice(range(20))
            if (self.state[s] != 'LONG' and self.state[s] != 'SHORT'):
              if choice == 1:
                print('LONG {}'.format(s))
                self.state[s] = 'LONG'
                signal = SignalEvent(1, ex, s, dt, 'LONG', 1.0)
                signals.append(signal)
              elif choice == 2:
                print('SHORT {}'.format(s))
                self.state[s] = 'SHORT'
                signal = SignalEvent(1, ex, s, dt, 'SHORT', 1.0)
                signals.append(signal)

            else:
              # For each instrument, we have 1/10 chance to exit a position
              if choice == 1 or choice == 2:
                print('EXIT {}'.format(s))
                self.state[s] = 'EXIT'
                signal = SignalEvent(1, ex, s, dt, 'EXIT', 1.0)
                signals.append(signal)

          if signals:
            events = SignalEvents(signals, id)
            self.events.put(events)