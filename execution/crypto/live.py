# Python

import os
import pdb
import time
import ccxt
import json
import logging
import datetime

from .execution import ExecutionHandler
from abc import ABCMeta, abstractmethod
from event import FillEvent, OrderEvent, BulkOrderEvent
from utils.helpers import from_standard_to_exchange_notation, convert_order_type

try:
    import Queue as queue
except ImportError:
    import queue

class LiveExecutionHandler(ExecutionHandler):
    """
    Handles order execution via the Interactive Brokers
    API, for use against accounts when trading live
    directly.
    """

    def __init__(self, events, configuration, exchanges):
        """
        Initialises the BitmexExecutionHandler instance.
        Parameters:
        events - The Queue of Event objects.
        """
        self.events = events
        self.exchanges = exchanges
        self.take_profit_gap = 0.5
        self.stop_loss_gap = 0.5
        self.fill_dict = {}
        self.order_id = 1
        self.logger = logging.getLogger('logger')

    def _error_handler(self, msg):
        """Handles the capturing of error messages"""
        # Currently no error handling.
        self.logger.error("Server Error: %s" % msg)

    def _reply_handler(self, msg):
        """Handles of server replies"""
        # Handle open order orderId processing

    def create_order(self, order_type, quantity, action):
        """Create an Order object (Market/Limit) to go long/short.
        order_type - 'MKT', 'LMT' for Market or Limit orders
        quantity - Integral number of assets to order
        action - 'BUY' or 'SELL'"""
        order = Order()
        order.m_orderType = order_type
        order.m_totalQuantity = quantity
        order.m_action = action
        return order

    def create_open_order_entry(self, msg):
        """
        Creates an entry in the Fill Dictionary that lists
        orderIds and provides security information. This is
        needed for the event-driven behaviour of the IB
        server message behaviour.
        """

    def create_fill(self, msg):
        """
        Handles the creation of the FillEvent that will be
        placed onto the events queue subsequent to an order
        being filled.
        """

    def create_market_order(self, exchange, symbol, type, side, amount, params):
        """
        Creates a market order on the bitmex exchange
        """
        order = exchange.create_order(symbol, side, amount, params)

        return order

    def create_limit_order(self, exchange, symbol, type, side, amount, price, params):
        """
        Creates a limit order on the bitmex exchange
        """
        order = exchange.create_order(symbol, side, amount, price, params)

        return order

    def execute_limit_order(self, event):
      exchange = event.exchange
      symbol = event.symbol
      order_type = event.order_type
      quantity = event.quantity
      direction = event.direction

      order = self.exchanges[exchange].create_order(symbol, order_type, direction, quantity)
      order_id = order['id']
      price = order['price']

      self.fill_dict[order] = {
        "symbol": symbol,
        "exchange": exchange,
        "direction": direction,
        "filled": False
      }

      if order['status'] == "closed" and self.fill_dict[order_id]["filled"] == False:
        filled = True
        fill_cost = price

      # Create a fill event object
      fill = FillEvent(
          datetime.datetime.utcnow(), symbol,
          exchange, filled, direction, fill_cost
      )

      # Make sure that multiple messages don't create
      # additional fills.
      self.fill_dict[order_id]["filled"] = True
      # Place the fill event onto the event queue
      self.events.put(fill)

      # symbol = event.symbol
    def execute_market_order(self, event):
      exchange = event.exchange
      symbol = from_standard_to_exchange_notation(exchange, event.symbol)
      order_type = event.order_type
      quantity = event.quantity
      direction = event.direction
      params = {}

      order = self.exchanges[event.exchange].create_order(symbol, order_type, direction, quantity, None, params)
      order_id = order['id']
      price = order['price']

      if order['status'] == "closed":
        filled = True

        self.fill_dict[order_id] = {
        "symbol": symbol,
        "exchange": exchange,
        "direction": direction,
        "filled": filled,
        }

        # Create a fill event object
        fill = FillEvent(
            datetime.datetime.utcnow(), symbol,
            exchange, filled, direction, price
        )

        # Place the fill event onto the event queue
        self.logger.info("ORDER FILLED")
        self.events.put(fill)
      else:
        self.logger.info("ORDER COULD NOT BE FILLED")

    def execute_take_profit(self, event):
      self.logger.info("Executing take profit order")
      self.logger.info("Price: {}".format(event.price))
      exchange = event.exchange
      symbol = from_standard_to_exchange_notation(exchange, event.symbol)
      order_type = convert_order_type(event.order_type)
      quantity = event.quantity
      direction = event.direction
      price = event.price
      params = { 'ordType': 'MarketIfTouched', 'stopPx': price, 'execInst': 'LastPrice,Close' }

      order = self.exchanges[event.exchange].create_order(symbol, order_type, direction, quantity, None, params)
      self.logger.info('TAKE PROFIT ORDER SET')

    def execute_stop_loss(self, event):
      self.logger.info("Executing stop loss order")
      self.logger.info("Price: {}".format(event.price))
      exchange = event.exchange
      symbol = from_standard_to_exchange_notation(exchange, event.symbol)
      order_type = convert_order_type(event.order_type)
      quantity = event.quantity
      direction = event.direction
      price = event.price
      params = { 'ordType': 'Stop', 'stopPx': price, 'execInst': 'LastPrice,Close'}

      order = self.exchanges[event.exchange].create_order(symbol, order_type, direction, quantity, None, params)
      self.logger.info('STOP LOSS ORDER SET')

    def execute_close_position(self, event):
      self.logger.info('Executing close {} position'.format(event.symbol))
      exchange = event.exchange
      symbol = from_standard_to_exchange_notation(exchange, event.symbol)
      params = { 'symbol': symbol }

      self.exchanges[exchange].private_post_order_closeposition(params)
      self.logger.info('CLOSE POSITION DONE')

    def execute_cancel_all_orders(self, event):
      self.logger.info('Executing close {} position'.format(event.symbol))
      exchange = event.exchange
      symbol = from_standard_to_exchange_notation(exchange, event.symbol)
      params = { 'symbol': symbol }

      self.exchanges[exchange].private_delete_order_all(params)
      self.logger.info('CANCEL ALL ORDERS')


    def execute_order(self, event):
      """
      Parameters:
      event - Contains an Event object with order information.
      """
      if event.type == 'ORDER':
        if event.order_type == "Market":
          self.execute_market_order(event)
        elif event.order_type == "TakeProfit":
          self.execute_take_profit(event)
        elif event.order_type == "StopLoss":
          self.execute_stop_loss(event)
        elif event.order_type == "Limit":
          self.execute_limit_order(event)
        elif event.order_type == "CancelAll":
          self.execute_cancel_all_orders(event)
        elif event.order_type == "ClosePosition":
          self.execute_close_position(event)

      self.order_id += 1









