from __future__ import print_function

from abc import ABCMeta, abstractmethod
import datetime
import os, os.path
import numpy as np
import pandas as pd

from event import MarketEvent

class DataHandler(object):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).
    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OHLCVI) for each symbol requested.
    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bar(self, symbol):
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        raise NotImplementedError("Should implement get_latest_bar_datetime()")

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        raise NotImplementedError("Should implement get_latest_bar_value()")

    @abstractmethod
    def get_latest_bar_values(self, symbol, val_type, N=1):
        """
        :return: Returns hte last N bar values from the latest_symbol list, or
            the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_values()")

    @abstractmethod
    def update_bars(self):
        """
        :return: Returns the latest bars to the bars_queue for each symbol in a
            tuple OHLCVI format: (datetime, open, high, low, close, volume,
            open interest).
        """
        raise NotImplementedError("Should implement update_bars()")

