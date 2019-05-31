#!/usr/bin/python
# -*- coding: utf-8 -*-

# portfolio.py
from __future__ import print_function

import os
import datetime
import functools
from math import floor

try:
    import Queue as queue
except ImportError:
    import queue

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfolio as pf
import seaborn as sns; sns.set()

from event import FillEvent, OrderEvent, BulkOrderEvent
from performance import create_sharpe_ratio, create_drawdowns
from utils import ceil_dt, from_exchange_to_standard_notation, from_standard_to_exchange_notation, truncate

class BitmexPortfolio(object):
    """
    The CryptoPortfolio is similar to the previous portfolio
    class. Instead of using the adjusted close data point, it uses
    the close datapoint
    """

    def __init__(self, data, events, configuration, exchanges):
        """
        Initialises the portfolio with data and an event queue.
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

        self.current_positions = self.construct_current_positions()
        self.all_positions = [ self.current_positions ]

        self.current_holdings = self.construct_current_holdings()
        self.all_holdings = [ self.current_holdings ]


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

        for p in positions:
          s = from_exchange_to_standard_notation('bitmex', p['symbol'])
          price = self.data.get_latest_bar_value('bitmex', s, "close") or 0
          btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', "close") or 0
          quantity = p['currentQty']
          if s in self.instruments:
            d['bitmex-{}'.format(s)] = quantity
            d['bitmex-{}-price'.format(s)] = price
            d['bitmex-{}-in-BTC'.format(s)] = quantity * price
            d['bitmex-{}-in-USD'.format(s)] = quantity * price * btc_price
            d['bitmex-{}-leverage'.format(s)] = 1
            d['total'] += quantity * price
            d['total-USD'] += quantity * price * btc_price


        print(d)

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
        latest_datetime = self.data.get_latest_bar_datetime('bitmex', self.instruments[0])

        # Update positions
        # ================
        dp = {}
        dp['datetime'] = latest_datetime

        for s in self.instruments:
          quantity = self.current_positions['bitmex-{}'.format(s)]
          price = self.current_positions['bitmex-{}-price'.format(s)]
          btc_price = self.current_positions['bitmex-BTC-price']
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

        print(dp)

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        # ===============
        dh = {}
        dh['datetime'] = latest_datetime
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['total']

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

        print(dh)

        for s in self.assets:
          self.current_holdings['bitmex-{}-fill'.format(s)] = 0

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

        for p in positions:
          s = from_exchange_to_standard_notation('bitmex', p['symbol'])
          price = self.data.get_latest_bar_value('bitmex', s, "close")
          btc_price = self.data.get_latest_bar_value('bitmex', 'XBTUSD', 'close')
          quantity = p['currentQty']
          if s in self.instruments:
            self.current_positions['bitmex-{}'.format(s)] = quantity
            self.current_positions['bitmex-BTC-price'] = btc_price
            self.current_positions['bitmex-{}-price'.format(s)] = price
            self.current_positions['bitmex-{}-in-BTC'.format(s)] = quantity * price
            self.current_positions['bitmex-{}-in-USD'.format(s)] = quantity * price * btc_price
            self.current_positions['bitmex-{}-leverage'.format(s)] = 1


    def rebalance_portfolio(self, signals):
        available_balance = self.current_holdings['total']
        total_strength = functools.reduce(lambda a,b : a.strength + b.strength, signals.events)
        exchange = 'bitmex'
        new_order_events = []
        cancel_orders_events = []
        events = []

        for (i, sig) in enumerate(signals.events):
          if sig.signal_type == "EXIT":
            price = self.data.get_latest_bar_value('bitmex', sig.symbol, "close")
            direction = 0
            current_quantity = self.current_positions['bitmex-{}'.format(sig.symbol)]
            target_allocation = direction * available_balance * sig.strength / total_strength
            target_quantity = floor(target_allocation / price)

            side = 'buy' if (target_quantity - current_quantity) > 0 else 'sell'
            quantity = abs(target_quantity - current_quantity)

            if (quantity == 0):
                continue

            close_position_order = OrderEvent(exchange, sig.symbol, 'Market', quantity, side, 1)
            cancel_other_orders = OrderEvent(exchange, sig.symbol, 'CancelAll')
            new_order_events.append(close_position_order)
            cancel_orders_events.append(cancel_other_orders)

          else:
            order_type = 'Market'
            price = self.data.get_latest_bar_value('bitmex', sig.symbol, "close")
            direction = { 'LONG': 1, 'SHORT': -1 }[sig.signal_type]
            current_quantity = self.current_positions['bitmex-{}'.format(sig.symbol)]
            target_allocation = direction * available_balance * sig.strength / total_strength
            target_quantity = floor(target_allocation / price)

            side = 'buy' if (target_quantity - current_quantity) > 0 else 'sell'
            quantity = abs(target_quantity - current_quantity)

            if (quantity == 0):
                continue

            order = OrderEvent(exchange, sig.symbol, order_type, quantity, side, 1)

            if side == 'buy':
              other_side = 'sell'
            elif side == 'sell':
              other_side = 'buy'

            if other_side == 'sell':
              stop_loss_stop_px = truncate(1.1 * price)
              take_profit_stop_px = truncate(0.9 * price)
            elif other_side == 'buy':
              stop_loss_stop_px = truncate(0.9 * price)
              take_profit_stop_px = truncate(1.1 * price)

            stop_loss_params = { 'stopPx': stop_loss_stop_px, 'execInst': 'LastPrice,Close' }
            stop_loss = OrderEvent(exchange, sig.symbol, 'Stop', None, side, 1, stop_loss_params)
            take_profit_params = { 'stopPx': take_profit_stop_px, 'execInst': 'LastPrice,Close' }
            take_profit = OrderEvent(exchange, sig.symbol, 'MarketIfTouched', None, side, 1, take_profit_params)
            cancel_other_orders = OrderEvent(exchange, sig.symbol, 'CancelAll')

            new_order_events += [order, stop_loss, take_profit]
            cancel_orders_events.append(cancel_other_orders)


        events = cancel_orders_events + new_order_events
        return events

    def generate_naive_order(self, signal):
        """
        Simply files an Order object as a constant quantity
        sizing of the signal object, without risk management or
        position sizing considerations.
        Parameters:
        signal - The tuple containing Signal information.
        """
        order = None
        exchange = signal.exchange
        symbol = signal.symbol
        direction = signal.signal_type
        market_quantity = 1
        current_quantity = self.current_positions['bitmex-{}'.format(symbol)]
        order_type = 'Market'

        if direction == 'LONG' and current_quantity == 0:
            order = OrderEvent(exchange, symbol, order_type, market_quantity, 'buy')
        if direction == 'SHORT' and current_quantity == 0:
            order = OrderEvent(exchange, symbol, order_type, market_quantity, 'sell')

        if direction == 'EXIT' and current_quantity > 0:
            order = OrderEvent(exchange, symbol, order_type, abs(current_quantity), 'sell')
        if direction == 'EXIT' and current_quantity < 0:
            order = OrderEvent(exchange, symbol, order_type, abs(current_quantity), 'buy')

        return order

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders
        based on the portfolio logic.
        """
        if event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event)
            self.events.put(order_event)


    def update_signals(self, events):
        order_events = self.rebalance_portfolio(events)
        for event in order_events:
          self.events.put(event)

    # ========================
    # POST-BACKTEST STATISTICS
    # ========================

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