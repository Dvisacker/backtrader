# Python
import os
import ccxt
import time
import json
import logging
import datetime

from .execution import ExecutionHandler
from abc import ABCMeta, abstractmethod
from event import FillEvent, OrderEvent, BulkOrderEvent
from utils.helpers import from_standard_to_exchange_notation

try:
    import Queue as queue
except ImportError:
    import queue

class SimulatedCryptoExchangeExecutionHandler(ExecutionHandler):
    """
    Handles order execution via the Interactive Brokers
    API, for use against accounts when trading live
    directly.
    """

    def __init__(self, data, events, configuration):
        """
        Initialises the BitmexExecutionHandler instance.
        Parameters:
        events - The Queue of Event objects.
        """
        self.data = data
        self.events = events
        self.take_profit_gap = 0.5
        self.stop_loss_gap = 0.5
        self.fill_dict = {}
        self.order_id = 1

        self.instruments = configuration.instruments

        self.upper_limit_orders = dict( (k,v) for k,v in [(e, {}) for e in self.instruments])
        for e in self.upper_limit_orders:
          self.upper_limit_orders[e] = dict((k,v) for k,v in [(s,{}) for s in self.instruments[e]])

        self.lower_limit_orders = dict((k,v) for k,v in [(e, {}) for e in self.instruments])
        for e in self.lower_limit_orders:
          self.lower_limit_orders[e] = dict((k,v) for k,v in [(s,{}) for s in self.instruments[e]])

    def fill_triggered_orders(self, event):
      for e in self.instruments:
        for s in self.instruments[e]:
          open_p = self.data.get_latest_bar_value('bitmex', s, 'open') or 0
          upper = self.upper_limit_orders[e][s]
          lower = self.lower_limit_orders[e][s]
          triggered_upper = [ upper[key] for (key, value) in upper.items() if key <= open_p ]
          triggered_lower = [ lower[key] for (key, value) in lower.items() if key >= open_p ]
          self.upper_limit_orders[e][s] = dict((key, value) for key, value in upper.items() if key > open_p )
          self.lower_limit_orders[e][s] = dict((key, value) for key, value in lower.items() if key < open_p )

          if triggered_upper:
            for o in triggered_upper[0]:
              quantity = o['quantity']
              direction = o['direction']
              order_type = o['type']

              fill = FillEvent(
                  datetime.datetime.utcnow(), s,
                  e, quantity, direction,
                  price=None, fee=None, leverage=None,
                  entry_price=None, fill_type=order_type
              )

              self.events.put(fill)

          if triggered_lower:
            for o in triggered_lower[0]:
              quantity = o['quantity']
              direction = o['direction']
              order_type = o['type']

              fill = FillEvent(
                  datetime.datetime.utcnow(), s,
                  e, quantity, direction,
                  price=None, fee=None, leverage=None,
                  entry_price=None, fill_type=order_type
              )

              self.events.put(fill)

    def open_take_profit_order(self, event):
      exchange = event.exchange
      symbol = event.symbol
      quantity = event.quantity
      direction = event.direction
      price = event.price
      order_type = event.order_type

      # Create a fill event object
      if direction == "buy":
        order = { 'exchange': exchange, 'symbol': symbol, 'quantity': quantity, 'direction': direction, 'type': order_type }
        if price not in self.lower_limit_orders[exchange][symbol]:
          self.lower_limit_orders[exchange][symbol][price] = [order]
        else:
          self.lower_limit_orders[exchange][symbol][price].append(order)

      if direction == "sell":
        order = { 'exchange': exchange, 'symbol': symbol, 'quantity': quantity, 'direction': direction, 'type': order_type }
        if price not in self.upper_limit_orders[exchange][symbol]:
          self.upper_limit_orders[exchange][symbol][price] = [order]
        else:
          self.upper_limit_orders[exchange][symbol][price].append(order)

    def open_stop_loss_order(self, event):
      exchange = event.exchange
      symbol = event.symbol
      quantity = event.quantity
      direction = event.direction
      order_type = event.order_type
      price = event.price

      # Create a fill event object
      if direction == "sell":
        order = { 'exchange': exchange, 'symbol': symbol, 'quantity': quantity, 'direction': direction, 'type': order_type }
        if price not in self.upper_limit_orders[exchange][symbol]:
          self.lower_limit_orders[exchange][symbol][price] = [order]
        else:
          self.lower_limit_orders[exchange][symbol][price].append(order)

      if direction == "buy":
        order = { 'exchange': exchange, 'symbol': symbol, 'quantity': quantity, 'direction': direction, 'type': order_type }
        if price not in self.lower_limit_orders[exchange][symbol]:
          self.upper_limit_orders[exchange][symbol][price] = [order]
        else:
          self.upper_limit_orders[exchange][symbol][price].append(order)

    def open_limit_order(self, event):
      pass

    def execute_market_order(self, event):
      exchange = event.exchange
      symbol = event.symbol
      quantity = event.quantity
      direction = event.direction
      fill_type = { 'buy': 'MarketBuy', 'sell': 'MarketSell'}[direction]

      # Create a fill event object
      fill = FillEvent(
          datetime.datetime.utcnow(), symbol,
          exchange, quantity, direction, price=None, fee=None, leverage=None,
          entry_price=None, fill_type=fill_type
      )

      # Place the fill event onto the event queue
      self.events.put(fill)

    def cancel_all_orders(self, event):
      exchange = event.exchange
      symbol = event.symbol
      self.upper_limit_orders[exchange][symbol] = {}
      self.lower_limit_orders[exchange][symbol] = {}

    def execute_close_position(self, event):
      exchange = event.exchange
      symbol = event.symbol
      quantity = event.quantity
      direction = event.direction

      # Create a fill event object
      fill = FillEvent(
          datetime.datetime.utcnow(), symbol,
          exchange, quantity, direction,
          price=None, fee=None, leverage=None,
          entry_price=None, fill_type='ClosePosition'
      )

      self.events.put(fill)

    def execute_order(self, event):
      """
      Parameters:
      event - Contains an Event object with order information.
      """
      if event.type == 'ORDER':
        if event.order_type == "Market":
          self.execute_market_order(event)
        elif event.order_type == "TakeProfit":
          self.open_take_profit_order(event)
        elif event.order_type == "StopLoss":
          self.open_stop_loss_order(event)
        elif event.order_type == "Limit":
          self.open_limit_order(event)
        elif event.order_type == "CancelAll":
          self.cancel_all_orders(event)
        elif event.order_type == "ClosePosition":
          self.execute_close_position(event)

      self.order_id += 1









