from __future__ import print_function

import ccxt
import os, os.path
import numpy as np
import pandas as pd
import asyncio

from cryptofeed.callback import Callback
from cryptofeed import FeedHandler
from cryptofeed.exchanges import Bitmex
from cryptofeed.defines import TRADES
from utils.aggregate import OHLCV

from threading import Thread
from abc import ABCMeta, abstractmethod
from datetime import datetime
from event import MarketEvent
from .datahandler import DataHandler
from utils.helpers import to_standard_notation, to_bitmex_notation, from_bitmex_notation, from_exchange_to_standard_notation, from_standard_to_exchange_notation

class LiveDataHandler(DataHandler):
    """
    LiveDataHandler
    """
    def __init__(self, events, configuration, exchanges):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form
        ’symbol.csv’, where symbol is a string in the list.
        :param events: The Event Queue.
        :param csv_dir: Absolute directory path to the CSV files.
        :param instruments: A list of symbol strings.
        """

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.latest_timestamp = None
        self.events = events
        self.exchanges = exchanges
        self.exchange_names = configuration.exchange_names
        self.instruments = configuration.instruments
        self.ohlcv_window = configuration.ohlcv_window
        self.continue_backtest = True

        for e in self.instruments:
          for s in self.instruments[e]:
            self.symbol_data[e] = {}
            self.latest_symbol_data[e] = {}

        for e in self.instruments:
          for s in self.instruments[e]:
            self.symbol_data[e][s] = []
            self.latest_symbol_data[e][s] = []


        self._initialize_exchange_data(self.exchanges)
        Thread(target = self._listen_to_exchange_data).start()

        # In case we have only one exchange
        if len(self.exchange_names) == 1:
          default = list(self.exchanges)[0]
          self.exchange = self.exchanges[default]

    def _date_parse(self, timestamp):
        """
        Parses timestamps into python datetime objects.
        """
        return datetime.fromtimestamp(int(timestamp))

    def _initialize_exchange_data(self, exchanges):
      print('Initializing exchange data')
      for e in self.instruments:
        for s in self.instruments[e]:
          sym = {
            "ADA/BTC": "ADAM19",
            "BCH/BTC": "BCHM19",
            "EOS/BTC": "EOSM19",
            "ETH/BTC": "ETHM19",
            "LTC/BTC": "LTCM19",
            "TRX/BTC": "TRXM19",
            "XRP/BTC": "XRPM19",
            "BTC/USD": "BTC/USD",
            "ETH/USD": "ETH/USD"
          }[s]


          ohlcv = self.exchanges[e].fetch_ohlcv(sym, '1m', params={ 'reverse': True })
          parser = lambda x : { 'time': x[0] / 1000, 'open': x[1], 'high': x[2], 'low': x[3], 'close': x[4], 'timestamp': datetime.fromtimestamp(x[0] / 1000) }
          parsed_ohlcv = list(map(parser, ohlcv))

          self.latest_symbol_data[e][s] = parsed_ohlcv
          self.symbol_data[e][s] = parsed_ohlcv

      # self.events.put(MarketEvent())


    def _listen_to_exchange_data(self):
      # Listen to exchange data is executed in a separate thread which requires a new event loop
      asyncio.set_event_loop(asyncio.new_event_loop())
      f = FeedHandler()
      symbols = []
      for e in self.instruments:
        if e == 'bitmex':
          for i in self.instruments[e]:
            start_time = self.symbol_data[e][i][-1]['time']
            s = from_standard_to_exchange_notation(e, i)
            symbols.append(s)

          f.add_feed(Bitmex(pairs=symbols, channels=[TRADES], callbacks={TRADES: OHLCV(Callback(self._handle_new_bar_bitmex), start_time=start_time, exchange='bitmex', instruments=self.instruments, window=self.ohlcv_window)}))

      f.run()


    async def _handle_new_bar_bitmex(self, data, timestamp):
      """
      Insert a new bar into the data feed
      """
      print('Handling new bar bitmex')
      if data and timestamp:
        self.events.put(MarketEvent(data, timestamp))


    def insert_new_bar_bitmex(self, data, timestamp):
        for instrument in list(data.keys()):
          tick = data[instrument]
          symbol = from_exchange_to_standard_notation('bitmex', instrument)
          if (tick['open'] == 0 or tick['close'] == 0):
            previous_tick = self.symbol_data['bitmex'][symbol][-1]
            tick['open'] = previous_tick['open']
            tick['close'] = previous_tick['close']
            tick['high'] = previous_tick['high']
            tick['low'] = previous_tick['low']
            tick['volume'] = 0
            tick['vwap'] = 0
            self.symbol_data['bitmex'][symbol].append(tick)
            self.latest_symbol_data['bitmex'][symbol].append(tick)
          else:
            self.symbol_data['bitmex'][symbol].append(tick)
            self.latest_symbol_data['bitmex'][symbol].append(tick)


    def _get_new_bar(self, exchange, symbol):
        """
        Returns the latest bar from the data feed.
        :return:
        """
        return self.symbol_data[exchange][symbol].pop(0)

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
            return bars_list[-1]['timestamp']

    def get_latest_bar_value(self, exchange, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_symbol_data[exchange][symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][val_type]

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
            return np.array([b[val_type] for b in bars_list])

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