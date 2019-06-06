#!/usr/bin/python
# -*- coding: utf-8 -*-

# portfolio.py
from __future__ import print_function

import os
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
import matplotlib.pyplot as plt
import pyfolio as pf
import seaborn as sns; sns.set()

from writer.mongo_writer import MongoWriter
from event import FillEvent, OrderEvent, BulkOrderEvent
from performance import create_sharpe_ratio, create_drawdowns
from utils import ceil_dt, from_exchange_to_standard_notation, from_standard_to_exchange_notation, truncate, get_precision

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

        self.current_positions = self.construct_current_positions()
        self.all_positions = [ self.current_positions ]
        self.current_holdings = self.construct_current_holdings()
        self.all_holdings = [ self.current_holdings ]

        self.db = MongoWriter()

    def construct_current_positions(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """

        d = dict( (k,v) for k, v in [(s, 0) for s in self.instruments] )
        d['datetime'] = self.start_date
        d['total'] = 0.0
        d['total-USD'] = 0.0

        positions = self.exchange.private_get_position()

        for s in self.instruments:
          price = self.data.get_latest_bar_value('bitmex', s, 'close') or 0
          btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')
          exchange_symbol = from_standard_to_exchange_notation('bitmex', s)
          quantity = positions[exchange_symbol]['currentQty'] if exchange_symbol in positions else 0

          d['bitmex-{}'.format(s)] = quantity
          d['bitmex-{}-price'.format(s)] = price
          d['bitmex-{}-in-BTC'.format(s)] = quantity * price
          d['bitmex-{}-in-USD'.format(s)] = quantity * price * btc_price
          d['bitmex-{}-leverage'.format(s)] = 1
          d['total'] += quantity * price
          d['total-USD'] += quantity * price * btc_price

        return d

    def construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = {}
        d['datetime'] = self.start_date

        response = self.exchange.fetch_balance()
        balance = response['total']['BTC']
        available_balance = response['total']['BTC']
        price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', "close")

        d['bitmex-BTC-available-balance'] = available_balance
        d['bitmex-BTC-balance'] = balance
        d['bitmex-BTC-price'] = price
        d['bitmex-BTC-value'] = balance * price
        d['bitmex-BTC-fill'] = ''
        d['total-USD'] = balance * price
        d['commission'] = 0

        return d

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for the current
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OHLCV).
        Makes use of a MarketEvent from the events queue.
        """
        print('Updating Time Index')
        latest_datetime = self.data.get_latest_bar_datetime('bitmex', self.instruments[0])

        # Update positions
        # ================
        dp = {}
        dp['datetime'] = latest_datetime
        dp['total'] = 0
        dp['total-USD'] = 0

        for s in self.instruments:
          quantity = self.current_positions['bitmex-{}'.format(s)]
          price = self.current_positions['bitmex-{}-price'.format(s)]
          btc_price = self.current_positions['bitmex-BTC/USD-price']
          dp['bitmex-{}'.format(s)] = self.current_positions['bitmex-{}'.format(s)]
          dp['bitmex-{}-price'.format(s)] = self.current_positions['bitmex-{}-price'.format(s)]
          dp['bitmex-{}-in-BTC'.format(s)] = self.current_positions['bitmex-{}-in-BTC'.format(s)]
          dp['bitmex-{}-in-USD'.format(s)] = self.current_positions['bitmex-{}-in-USD'.format(s)]

          if 'bitmex-{}-leverage'.format(s) in self.current_positions:
            dp['bitmex-{}-leverage'.format(s)] = self.current_positions['bitmex-{}-leverage'.format(s)]
          else:
            dp['bitmex-{}-leverage'.format(s)] = 0

          dp['total'] += quantity * price
          dp['total-USD'] += quantity * price * btc_price

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        # ===============
        dh = {}
        dh['datetime'] = latest_datetime
        dh['commission'] = self.current_holdings['commission']
        dh['total-USD'] = 0

        for s in self.assets:
          price = self.data.get_latest_bar_value('bitmex', '{}/USD'.format(s), "close")
          balance = self.current_holdings['bitmex-{}-balance'.format(s)]
          available_balance = self.current_holdings['bitmex-{}-available-balance'.format(s)]

          dh['bitmex-{}-available-balance'.format(s)] = available_balance
          dh['bitmex-{}-balance'.format(s)] = balance
          dh['bitmex-{}-price'.format(s)] = price
          dh['bitmex-{}-value'.format(s)] = balance * price
          dh['bitmex-{}-fill'.format(s)] = self.current_holdings['bitmex-{}-fill'.format(s)]
          dh['total-USD'] += balance * price

        dh['commission'] += 0.0

        # Append the current holdings
        self.all_holdings.append(dh)

        for s in self.assets:
          self.current_holdings['bitmex-{}-fill'.format(s)] = 0

        self.write(dp, dh)

    # ======================
    # FILL/POSITION HANDLING
    # ======================
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings
        from a FillEvent.
        """
        if event.type == 'FILL':
          self.update_positions_from_fill(event)
          self.update_holdings_from_fill(event)

    def update_holdings_from_fill(self, fill):
        """
        Takes a Fill object and updates the holdings matrix to
        reflect the holdings value
        Parameters:
        fill - The Fill object to update the holdings with
        """
        symbol = fill.symbol
        data = self.exchange.fetch_balance()
        balances = data['total']

        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        balance = balances['BTC']
        available_balance = balances['BTC']
        price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', "close")

        # Update holdings list with new quantities
        self.current_holdings['bitmex-BTC-available-balance'] = available_balance
        self.current_holdings['bitmex-BTC-balance'] = balance
        self.current_holdings['bitmex-BTC-price'] = price
        self.current_holdings['bitmex-BTC-value'] = balance * price
        self.current_holdings['bitmex-BTC-fill'] = fill.direction
        self.current_holdings['commission'] += fill.commission


    def update_positions_from_fill(self, fill):
        """
        Takes a Fill object and updates the position matrix to
        reflect the new position.
        Parameters:
        fill - The Fill object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        positions = self.exchange.private_get_position()

        for s in self.instruments:
          price = self.data.get_latest_bar_value('bitmex', s, 'close') or 0
          btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')
          exchange_symbol = from_standard_to_exchange_notation('bitmex', s)
          quantity = positions[exchange_symbol]['currentQty'] if exchange_symbol in positions else 0

          self.current_positions['bitmex-{}'.format(s)] = quantity
          self.current_positions['bitmex-BTC-price'] = btc_price
          self.current_positions['bitmex-{}-price'.format(s)] = price
          self.current_positions['bitmex-{}-in-BTC'.format(s)] = quantity * price
          self.current_positions['bitmex-{}-in-USD'.format(s)] = quantity * price * btc_price
          self.current_positions['bitmex-{}-leverage'.format(s)] = 1

    def rebalance_portfolio_2(self, signals):
        """
        Rebalances the portfolio based on an array of signals. The amount of funds allocated to each
        signal depends on the relative strength given to each signal by the strategy module.
        Parameters:
        signals - Array of signal events
        """
        available_balance = self.current_holdings['bitmex-BTC-available-balance']
        total_strength = len(signals.events)
        exchange = 'bitmex'
        new_order_events = []
        cancel_orders_events = []
        events = []

        for sig in signals.events:
          sig.print_signal()

          price = self.data.get_latest_bar_value('bitmex', sig.symbol, "close")
          if not price:
            continue

          if sig.signal_type == "EXIT":
            cancel_open_orders = OrderEvent(exchange, sig.symbol, 'CancelAll')
            close_position_order = OrderEvent(exchange, sig.symbol, 'ClosePosition')
            cancel_orders_events.append(cancel_open_orders)
            new_order_events.append(close_position_order)
          else:
            direction = { 'LONG': 1, 'SHORT': -1 }[sig.signal_type]
            target_allocation = direction * available_balance * sig.strength / total_strength
            current_quantity = self.current_positions['bitmex-{}'.format(sig.symbol)]
            target_quantity = floor(target_allocation / price)

            side = 'buy' if (target_quantity - current_quantity) > 0 else 'sell'
            quantity = abs(target_quantity - current_quantity)

            if (quantity == 0):
                continue

            order = OrderEvent(exchange, sig.symbol, 'Market', quantity, side, 1)
            precision = get_precision(sig.symbol)

            if side == 'buy':
              other_side = 'sell'
              stop_loss_stop_px = truncate(0.9 * price, precision)
              take_profit_stop_px = truncate(1.1 * price, precision)
            elif side == 'sell':
              other_side = 'buy'
              stop_loss_stop_px = truncate(1.1 * price, precision)
              take_profit_stop_px = truncate(0.9 * price, precision)

            stop_loss_params = { 'stopPx': stop_loss_stop_px, 'execInst': 'LastPrice,Close' }
            stop_loss = OrderEvent(exchange, sig.symbol, 'Stop', None, other_side, 1, stop_loss_params)
            take_profit_params = { 'stopPx': take_profit_stop_px, 'execInst': 'LastPrice,Close' }
            take_profit = OrderEvent(exchange, sig.symbol, 'MarketIfTouched', None, other_side, 1, take_profit_params)
            cancel_other_orders = OrderEvent(exchange, sig.symbol, 'CancelAll')

            new_order_events += [order, stop_loss, take_profit]
            cancel_orders_events.append(cancel_other_orders)

        events = cancel_orders_events + new_order_events
        return events


    def rebalance_portfolio(self, signals):
        """
        Rebalances the portfolio based on an array of signals. The amount of funds allocated to each
        signal depends on the relative strength given to each signal by the strategy module.
        Parameters:
        signals - Array of signal events
        """
        available_balance = self.current_holdings['bitmex-BTC-available-balance']
        exchange = 'bitmex'
        new_order_events = []
        cancel_orders_events = []
        events = []
        default_position_size = self.default_position_size

        for sig in signals.events:
          sig.print_signal()
          price = self.data.get_latest_bar_value('bitmex', sig.symbol, "close")
          if not price:
            continue

          if sig.signal_type == "EXIT":
            cancel_open_orders = OrderEvent(exchange, sig.symbol, 'CancelAll')
            close_position_order = OrderEvent(exchange, sig.symbol, 'ClosePosition')
            cancel_orders_events.append(cancel_open_orders)
            new_order_events.append(close_position_order)
          else:
            direction = { 'LONG': 1, 'SHORT': -1 }[sig.signal_type]
            target_allocation = direction * default_position_size * sig.strength
            current_quantity = self.current_positions['bitmex-{}'.format(sig.symbol)]
            target_quantity = floor(target_allocation / price)
            side = 'buy' if (target_quantity - current_quantity) > 0 else 'sell'
            quantity = abs(target_quantity - current_quantity)

            if (target_allocation > available_balance):
                continue

            if (quantity == 0):
                continue

            order = OrderEvent(exchange, sig.symbol, 'Market', quantity, side, 1)
            precision = get_precision(sig.symbol)

            if side == 'buy':
              other_side = 'sell'
              stop_loss_stop_px = truncate(0.9 * price, precision)
              take_profit_stop_px = truncate(1.1 * price, precision)
            elif side == 'sell':
              other_side = 'buy'
              stop_loss_stop_px = truncate(1.1 * price, precision)
              take_profit_stop_px = truncate(0.9 * price, precision)

            stop_loss_params = { 'stopPx': stop_loss_stop_px, 'execInst': 'LastPrice,Close' }
            stop_loss = OrderEvent(exchange, sig.symbol, 'Stop', None, other_side, 1, stop_loss_params)
            take_profit_params = { 'stopPx': take_profit_stop_px, 'execInst': 'LastPrice,Close' }
            take_profit = OrderEvent(exchange, sig.symbol, 'MarketIfTouched', None, other_side, 1, take_profit_params)
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
        order_events = self.rebalance_portfolio(events)
        for event in order_events:
          self.events.put(event)

    # ========================
    # POST-BACKTEST STATISTICS
    # ========================

    def write(self, current_positions, current_holdings):
        """
        """
        print('Writing positions: {}'.format(current_positions))
        print('Writing holdings: {}'.format(current_holdings))
        self.db.insert_positions(current_positions)
        self.db.insert_holdings(current_holdings)

    def create_equity_curve_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings
        list of dictionaries.
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0+curve['returns']).cumprod()
        self.equity_curve = curve

    def print_summary_stats(self):
        """
        Print a list of summary statistics for the portfolio.
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]

        return stats


    def output_summary_stats(self, backtest_result_dir):
        """
        Creates a list of summary statistics for the portfolio.
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]

        # We output both to the most recent backtest folder and to a backtest timestamped folder
        self.equity_curve.to_csv(os.path.join(self.result_dir, 'equity.csv'))
        self.equity_curve.to_csv(os.path.join(backtest_result_dir, 'equity.csv'))
        return stats

    def output_summary_stats_and_graphs(self, backtest_result_dir):
        """
        Creates a list of summary statistics and plots
        performance graphs
        """

        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]

        self.equity_curve.to_csv(os.path.join(self.result_dir, 'equity.csv'))
        self.equity_curve.to_csv(os.path.join(backtest_result_dir, 'equity.csv'))

        returns = self.equity_curve['returns']
        equity_curve = self.equity_curve['equity_curve']
        drawdown = self.equity_curve['drawdown']

        # Plot three charts: Equity curve,
        # period returns, drawdowns
        fig = plt.figure(figsize=(15,10))
        # Set the outer colour to white
        fig.patch.set_facecolor('white')
        # Plot the equity curve
        ax1 = fig.add_subplot(311, ylabel='Portfolio value, %')
        equity_curve.plot(ax=ax1, color="blue", lw=2.)
        plt.grid(True)

        # Plot the returns
        ax2 = fig.add_subplot(312, ylabel='Period returns, %')
        returns.plot(ax=ax2, color="black", lw=2.)
        plt.grid(True)

        # Plot the returns
        ax3 = fig.add_subplot(313, ylabel='Drawdowns, %')
        drawdown.plot(ax=ax3, color="red", lw=2.)
        plt.grid(True)

        # pf.show_perf_stats(returns)
        # pf.show_worst_drawdown_periods(returns)

        plt.figure(figsize = (15, 10))
        pf.plot_drawdown_underwater(returns).set_xlabel('Date')

        plt.figure(figsize = (15, 10))
        pf.plot_drawdown_periods(returns, top=5).set_xlabel('Date')

        plt.figure(figsize = (15, 10))
        pf.plot_returns(returns).set_xlabel('Date')

        plt.figure(figsize = (15, 10))
        pf.plot_return_quantiles(returns).set_xlabel('Timeframe')

        plt.figure(figsize = (15, 10))
        pf.plot_monthly_returns_dist(returns).set_xlabel('Returns')

        plt.figure(figsize = (15, 10))
        pf.plot_rolling_volatility(returns, rolling_window=30).set_xlabel('date')

        plt.figure(figsize = (15, 10))
        pf.plot_rolling_sharpe(returns, rolling_window=30).set_xlabel('Date')

        pf.create_returns_tear_sheet(returns)
        return stats