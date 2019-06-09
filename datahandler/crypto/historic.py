from __future__ import print_function

from abc import ABCMeta, abstractmethod
from datetime import datetime
import os, os.path
import numpy as np
import pandas as pd

from event import MarketEvent
from utils.helpers import get_data_file, get_ohlcv_window
from .datahandler import DataHandler


class HistoricCSVCryptoDataHandler(DataHandler):
    """
    HistoricCSVCryptoDataHandler is designed to read CSV files for each requested
    symbol from disk and provide an interface to obtain the "latest" bar in
    a manner identical to a live trading interface.
    """
    def __init__(self, events, configuration):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form
        ’symbol.csv’, where symbol is a string in the list.
        :param events: The Event Queue.
        :param csv_dir: Absolute directory path to the CSV files.
        :param instruments: A list of symbol strings.
        """
        self.events = events
        self.csv_dir = configuration.csv_dir
        self.period = get_ohlcv_window(configuration.ohlcv_window)
        self.feeds = configuration.feeds
        self.start_date = configuration.start_date

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True

        self._open_convert_csv_files()

    def _date_parse(self, timestamp):
        """
        Parses timestamps into python datetime objects.
        """
        return datetime.fromtimestamp(int(timestamp))

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        For this handler it will be assumed that the data is
        taken from Yahoo. Thus its format will be respected.
        """
        comb_index = None
        for e in self.feeds:
          self.symbol_data[e] = {}
          self.latest_symbol_data[e] = {}

          for s in self.feeds[e]:
            csv_file = get_data_file(e, s, self.period)
            # Load the CSV
            df = pd.read_csv(
                os.path.join(self.csv_dir, csv_file),
                parse_dates = True,
                date_parser=self._date_parse,
                header=0,
                sep=',',
                index_col=0,
                names=['datetime', 'open', 'high', 'low', 'close', 'volume', 'id', 'date']
            )

            df.dropna(inplace=True)
            df['returns'] = df['close'].pct_change()

            # Truncate the data according to start_date
            self.symbol_data[e][s] = df.sort_index().ix[self.start_date:]

            if comb_index is None:
                comb_index = self.symbol_data[e][s].index
            else:
                comb_index.union(self.symbol_data[e][s].index)

            self.latest_symbol_data[e][s] = []
            self.symbol_data[e][s] = self.symbol_data[e][s].reindex(index=comb_index, method='pad').iterrows()

            for i in range(300):
              bar = next(self._get_new_bar(e, s))
              if bar is not None:
                  self.latest_symbol_data[e][s].append(bar)



        # print(self.latest_symbol_data['bitmex']['BTC/USD'])


    def _get_new_bar(self, exchange, symbol):
        """
        Returns the latest bar from the data feed.
        :return:
        """
        for b in self.symbol_data[exchange][symbol]:
            yield b

    def get_latest_bar(self, exchange, symbol):
        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_symbol_data[exchange][symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, exchange, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[exchange][symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, exchange, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        try:
            bars_list = self.latest_symbol_data[exchange][symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, exchange, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_symbol_data[exchange]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

    def get_latest_bars_values(self, exchange, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        try:
            bars_list = self.get_latest_bars(exchange, symbol, N)
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return np.array([getattr(b[1], val_type) for b in bars_list])

    def get_asset_value(self, exchange, asset_symbol):
        """
        Returns the asset value in dollars.
        If the given symbol is BTC, it will return the value of BTC/USD
        """
        try:
          instrument_symbol = asset_symbol + "/USD"
          bars_list = self.get_latest_bars(exchange, instrument_symbol, 1)
        except KeyError:
          print("That symbol is not available in the historical data set.")
          raise
        else:
          return np.array([getattr(b[1], 'close') for b in bars_list])

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for e in self.feeds:
          for s in self.feeds[e]:
            try:
                bar = next(self._get_new_bar(e, s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[e][s].append(bar)
        if self.continue_backtest:
            self.events.put(MarketEvent())  # Bug! When except occur, it means
            # the backtest should be terminated. So in this time, we should not
            # generate a MarketEvent again.

        # self.events.put(MarketEvent())  Bug! When except occur, it means
        # the backtest should be terminated. So in this time, we should not
        # generate a MarketEvent again.
