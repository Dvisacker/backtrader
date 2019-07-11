#!/usr/bin/python
# -*- coding: utf-8 -*-

# portfolio.py
import os
import pdb
import logging
import datetime
import functools
from math import floor
from pathlib import Path

try:
    import Queue as queue
except ImportError:
    import queue

import numpy as np
import pandas as pd
import webbrowser
import matplotlib.pyplot as plt
import pyfolio as pf
import seaborn as sns; sns.set()
import matplotlib.gridspec as gridspec

from db.mongo_handler import MongoHandler
from event import FillEvent, OrderEvent, BulkOrderEvent
from performance import create_sharpe_ratio, create_drawdowns
from utils import ceil_dt, from_exchange_to_standard_notation, from_standard_to_exchange_notation, truncate, get_precision
from utils.helpers import plot, move_figure, compute_all_indicators

from stats.trades import generate_trade_stats

class BitmexPortfolio(object):
    """
    The CryptoPortfolio is similar to the previous portfolio
    class. Instead of using the adjusted close data point, it uses
    the close datapoint
    """

    def __init__(self, data, events, configuration, exchanges):
        """
        Initializes the portfolio with data and an event queue.
        Also includes a starting datetime index and initial capital
        (USD unless otherwise stated).
        Parameters:
        data - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """

        if 'bitmex' not in exchanges:
          raise AssertionError

        self.data = data
        self.events = events
        self.exchange = exchanges['bitmex']
        self.instruments = configuration.instruments['bitmex']
        self.assets = configuration.assets['bitmex']
        self.start_date = configuration.start_date
        self.result_dir = configuration.result_dir
        self.default_position_size = configuration.default_position_size
        self.indicators = configuration.indicators
        self.save_to_db = configuration.save_to_db
        self.default_leverage = configuration.default_leverage

        self.current_portfolio = self.construct_current_portfolios()
        self.all_portfolios = []
        self.all_transactions = []

        self.db = MongoHandler()
        self.legends_added = False

        self.take_profit_gap = configuration.take_profit_gap
        self.stop_loss_gap = configuration.stop_loss_gap
        self.use_stops = configuration.use_stops

    def construct_current_portfolios(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """

        d = {}
        d['total'] = 0.0
        d['total-in-USD'] = 0.0

        position_array = self.exchange.private_get_position()
        positions = { p['symbol']: p for p in position_array }
        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')

        response = self.exchange.fetch_balance()
        btc_balance = response['total']['BTC']
        available_btc_balance = response['total']['BTC']

        for s in self.instruments:
          price = self.data.get_latest_bar_value('bitmex', s, 'close') or 0

          exchange_symbol = from_standard_to_exchange_notation('bitmex', s)
          quantity = positions[exchange_symbol]['currentQty'] if exchange_symbol in positions else 0

          d['bitmex-{}'.format(s)] = quantity
          d['bitmex-{}-price'.format(s)] = price
          d['bitmex-{}-position'.format(s)] = quantity
          d['bitmex-{}-position-in-BTC'.format(s)] = quantity * price
          d['bitmex-{}-position-in-USD'.format(s)] = quantity * price * btc_price
          d['bitmex-{}-leverage'.format(s)] = self.default_leverage
          d['bitmex-{}-fill'.format(s)] = ''
          d['total'] += quantity * price
          d['total-in-USD'] += quantity * price * btc_price

        d['bitmex-BTC-available-balance'] = available_btc_balance
        d['bitmex-BTC-balance'] = btc_balance
        d['bitmex-BTC-price'] = btc_price
        d['total'] += btc_balance
        d['total-in-USD'] += btc_balance * btc_price
        d['fee'] = 0

        return d

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for the current
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OHLCV).
        Makes use of a MarketEvent from the events queue.
        """
        latest_datetime = self.data.get_latest_bar_datetime('bitmex', self.instruments[0])

        # Update positions
        # ================
        df = {}
        df['datetime'] = latest_datetime
        df['bitmex-total-position-in-BTC'] = 0
        df['bitmex-total-position-in-USD'] = 0

        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')

        for s in self.instruments:
          quantity = self.current_portfolio['bitmex-{}-position'.format(s)]
          price = self.data.get_latest_bar_value('bitmex', s, 'close')
          df['bitmex-{}-price'.format(s)] = price
          df['bitmex-{}-position'.format(s)] = quantity
          df['bitmex-{}-position-in-BTC'.format(s)] = quantity * price
          df['bitmex-{}-position-in-USD'.format(s)] = quantity * price * btc_price
          df['bitmex-{}-fill'.format(s)] = self.current_portfolio['bitmex-{}-fill'.format(s)]

          if 'bitmex-{}-leverage'.format(s) in self.current_portfolio:
            df['bitmex-{}-leverage'.format(s)] = self.current_portfolio['bitmex-{}-leverage'.format(s)]
          else:
            df['bitmex-{}-leverage'.format(s)] = 0

          df['bitmex-total-position-in-BTC'] += quantity * price
          df['bitmex-total-position-in-USD'] += quantity * price * btc_price

        # Update holdings
        # ===============
        df['fee'] = self.current_portfolio['fee']
        df['total'] = 0
        df['total-in-USD'] = 0

        for s in self.assets:
          price = self.data.get_latest_bar_value('bitmex', '{}/USD'.format(s), "close")
          balance = self.current_portfolio['bitmex-{}-balance'.format(s)]
          available_balance = self.current_portfolio['bitmex-{}-available-balance'.format(s)]

          df['bitmex-{}-available-balance'.format(s)] = available_balance
          df['bitmex-{}-balance'.format(s)] = balance
          df['bitmex-{}-balance-in-USD'.format(s)] = balance * price
          df['bitmex-{}-price'.format(s)] = price
          df['total'] += balance
          df['total-in-USD'] += balance * price

        # Append the current holdings
        self.all_portfolios.append(df)

        for s in self.assets:
          self.current_portfolio['bitmex-{}-fill'.format(s)] = ''

        if self.save_to_db:
          self.write_to_db(df)

    # ======================
    # FILL/POSITION HANDLING
    # ======================
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings
        from a FillEvent.
        """
        if event.type == 'FILL' and (event.fill_type == 'MarketBuy' or event.fill_type == 'MarketSell'):
          self.update_portfolio_from_fill(event)
        elif event.type == 'FILL' and event.fill_type == 'ClosePosition':
          self.update_portfolio_from_exit(event)
        elif event.type == 'FILL' and event.fill_type == 'StopLoss':
          self.update_portfolio_from_exit(event)
        elif event.type == 'FILL' and event.fill_type == 'TakeProfit':
          self.update_portfolio_from_exit(event)


    def update_portfolio_from_fill(self, fill):
        """
        Takes a Fill object and updates the position matrix to
        reflect the new position.
        Parameters:
        fill - The Fill object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        latest_datetime = self.data.get_latest_bar_datetime('bitmex', self.instruments[0])
        direction = { 'buy': 1, 'sell': -1 }[fill.direction]
        symbol = fill.symbol
        quantity = fill.quantity
        fee_rate = fill.fee
        fill_type = fill.fill_type

        data = self.exchange.fetch_balance()
        balances = data['total']
        btc_balance = balances['BTC']
        btc_available_balance = balances['BTC']

        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')
        previous_position = self.current_portfolio['bitmex-{}-position'.format(symbol)]
        new_position = previous_position + direction * quantity
        entry_price = self.data.get_latest_bar_value('bitmex', symbol, 'close')

        btc_value = entry_price * quantity
        btc_fee = btc_value * fee_rate
        leverage = self.default_leverage

        self.current_portfolio['bitmex-{}-position'.format(symbol)] += direction * quantity
        self.current_portfolio['bitmex-{}-price'.format(symbol)] = entry_price
        self.current_portfolio['bitmex-{}-position-in-BTC'.format(symbol)] = direction * btc_value
        self.current_portfolio['bitmex-{}-position-in-USD'.format(symbol)] = direction * btc_value * btc_price
        self.current_portfolio['bitmex-{}-leverage'.format(symbol)] = leverage
        self.current_portfolio['bitmex-{}-fill'.format(symbol)] = fill_type
        self.current_portfolio['bitmex-total-position-in-BTC'] += direction * btc_value * btc_price
        self.current_portfolio['bitmex-total-position-in-USD'] += direction * btc_value

        # Update holdings list with new quantities
        self.current_portfolio['bitmex-BTC-balance'] -= btc_fee
        self.current_portfolio['bitmex-{}-entry-price'.format(symbol)] = entry_price
        self.current_portfolio['bitmex-BTC-available-balance'] = btc_available_balance - (abs(new_position) - abs(previous_position)) * (entry_price / leverage) - btc_fee
        self.current_portfolio['bitmex-BTC-price'] = btc_price
        self.current_portfolio['bitmex-BTC-balance-in-USD'] = (btc_balance - btc_fee) * btc_price
        self.current_portfolio['total'] = (btc_balance - btc_fee)
        self.current_portfolio['total-in-USD'] = (btc_balance - btc_fee) * btc_price
        self.current_portfolio['fee'] += btc_fee

        txn = {}
        txn['datetime'] = latest_datetime
        txn['amount'] = direction * quantity
        txn['price'] = entry_price * btc_price
        txn['txn_dollars'] = direction * entry_price * btc_price * quantity
        txn['symbol'] = symbol

        # pdb.set_trace()
        self.all_transactions.append(txn)

    def update_portfolio_from_exit(self, fill):
        symbol = fill.symbol
        fee_rate = fill.fee
        fill_type = fill.fill_type

        latest_datetime = self.data.get_latest_bar_datetime('bitmex', self.instruments[0])
        data = self.exchange.fetch_balance()
        balances = data['total']
        btc_balance = balances['BTC']
        btc_available_balance = balances['BTC']

        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')
        entry_price = self.current_portfolio['bitmex-{}-entry-price'.format(symbol)]
        price = self.data.get_latest_bar_value('bitmex', symbol, 'close') or 0
        quantity = self.current_portfolio['bitmex-{}-position'.format(symbol)]
        direction = -1 if quantity > 0 else 1
        leverage = self.default_leverage

        # We distinguish between the different contracts
        if (symbol == 'BTC/USD'):
          btc_value = quantity / price
          btc_fee = btc_value * fee_rate
          self.current_portfolio['bitmex-{}-position'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-price'.format(symbol)] = price
          self.current_portfolio['bitmex-{}-position-in-BTC'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-position-in-USD'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-leverage'.format(symbol)] = leverage
          self.current_portfolio['bitmex-{}-fill'.format(symbol)] = fill_type
          self.current_portfolio['bitmex-total-position-in-BTC'] -= btc_value
          self.current_portfolio['bitmex-total-position-in-USD'] -= btc_value * btc_price
          self.current_portfolio['bitmex-BTC-balance'] = btc_balance
          self.current_portfolio['bitmex-BTC-available-balance'] = btc_available_balance
          self.current_portfolio['bitmex-BTC-price'] = btc_price
          self.current_portfolio['bitmex-BTC-balance-in-USD'] = btc_balance * btc_price

          # here, total, totalUSD are equal to the bitmex-BTC-Balance and bitmex-BTC-balance-in-USD fields
          self.current_portfolio['total'] = btc_balance
          self.current_portfolio['total-in-USD'] = btc_balance * btc_price
          self.current_portfolio['fee'] += btc_fee

        else:
          btc_value = quantity * price
          btc_fee = btc_value * fee_rate
          self.current_portfolio['bitmex-{}-position'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-price'.format(symbol)] = price
          self.current_portfolio['bitmex-{}-position-in-BTC'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-position-in-USD'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-leverage'.format(symbol)] = leverage
          self.current_portfolio['bitmex-{}-fill'.format(symbol)] = fill_type
          self.current_portfolio['bitmex-total-position-in-BTC'] -= btc_value
          self.current_portfolio['bitmex-total-position-in-USD'] -= btc_value * btc_price
          self.current_portfolio['bitmex-BTC-balance'] = btc_balance
          self.current_portfolio['bitmex-BTC-available-balance'] = btc_available_balance
          self.current_portfolio['bitmex-BTC-price'] = btc_price
          self.current_portfolio['bitmex-BTC-balance-in-USD'] = btc_balance * btc_price

          # here, total, totalUSD are equal to the bitmex-BTC-Balance and bitmex-BTC-balance-in-USD fields
          self.current_portfolio['total'] = btc_balance
          self.current_portfolio['total-in-USD'] = btc_balance * btc_price
          self.current_portfolio['fee'] += btc_fee

        txn = {}
        txn['datetime'] = latest_datetime
        txn['amount'] = direction * abs(quantity)
        txn['price'] = price * btc_price
        txn['txn_dollars'] = direction * entry_price * btc_price * abs(quantity)
        txn['symbol'] = symbol

        self.all_transactions.append(txn)

    def rebalance_portfolio(self, signals):
        """
        Rebalances the portfolio based on an array of signals. The amount of funds allocated to each
        signal depends on the relative strength given to each signal by the strategy module.
        Parameters:
        signals - Array of signal events
        """
        available_balance = self.current_portfolio['bitmex-BTC-available-balance']
        exchange = 'bitmex'
        new_order_events = []
        cancel_orders_events = []
        events = []
        default_position_size = self.default_position_size

        for sig in signals.events:
          sig.print_signal()
          price = self.data.get_latest_bar_value('bitmex', sig.symbol, "close")
          if not price:
            # Might want to throw an error here
            continue

          if sig.signal_type == "EXIT":
            cancel_open_orders = OrderEvent(exchange, sig.symbol, 'CancelAll')
            close_position_order = OrderEvent(exchange, sig.symbol, 'ClosePosition')
            cancel_orders_events.append(cancel_open_orders)
            new_order_events.append(close_position_order)
          else:
            direction = { 'LONG': 1, 'SHORT': -1 }[sig.signal_type]
            target_allocation = direction * default_position_size * sig.strength
            current_quantity = self.current_portfolio['bitmex-{}'.format(sig.symbol)]
            target_quantity = floor(target_allocation / price)
            side = 'buy' if (target_quantity - current_quantity) > 0 else 'sell'
            quantity = abs(target_quantity - current_quantity)

            print('TARGET ALLOCATION: {}'.format(target_allocation))
            print('PRICE: {}'.format(price))
            print('CURRENT QUANTITY: {}'.format(current_quantity))
            print('POSITION QUANTITY: {} for {}'.format(target_quantity, sig.symbol))

            if (target_allocation > available_balance):
                # Might want to throw an error here
                continue

            if (quantity == 0):
              # Might want to throw an error here
                continue

            order = OrderEvent(exchange, sig.symbol, 'Market', quantity, side, 1)
            precision = get_precision(sig.symbol)

            if side == 'buy':
              other_side = 'sell'
              stop_loss_px = truncate((1 - self.stop_loss_gap) * price, precision)
              take_profit_px = truncate((1 + self.take_profit_gap) * price, precision)
            elif side == 'sell':
              other_side = 'buy'
              stop_loss_px = truncate((1 + self.stop_loss_gap) * price, precision)
              take_profit_px = truncate((1 - self.take_profit_gap) * price, precision)

            stop_loss = OrderEvent(exchange, sig.symbol, 'StopLoss', quantity, other_side, 1, stop_loss_px)
            take_profit = OrderEvent(exchange, sig.symbol, 'TakeProfit', quantity, other_side, 1, take_profit_px)
            cancel_other_orders = OrderEvent(exchange, sig.symbol, 'CancelAll')

            new_order_events += [order, stop_loss, take_profit]
            cancel_orders_events.append(cancel_other_orders)

        events = cancel_orders_events + new_order_events
        return events

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders
        based on the portfolio logic.
        """
        pass


    def update_signals(self, events):
        """
        Acts on a SignalEvents to generate new orders
        based on the portfolio logic.
        """
        order_events = self.rebalance_portfolio(events)
        for event in order_events:
          self.events.put(event)

    # ========================
    # POST-BACKTEST STATISTICS
    # ========================

    def write_to_db(self, current_portfolio):
        """
        Saves the position and holdings updates to a database or to a file
        """
        self.db.insert_portfolio(current_portfolio)