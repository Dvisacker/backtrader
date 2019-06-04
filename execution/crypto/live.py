# Python
from __future__ import print_function

import datetime
import time
import os
import ccxt
import json

from .execution import ExecutionHandler
from abc import ABCMeta, abstractmethod
from event import FillEvent, OrderEvent, BulkOrderEvent
from utils.helpers import from_standard_to_exchange_notation

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

    def _error_handler(self, msg):
        """Handles the capturing of error messages"""
        # Currently no error handling.
        print("Server Error: %s" % msg)

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
      print("IN EXECUTE MARKET ORDER")
      exchange = event.exchange
      symbol = from_standard_to_exchange_notation(exchange, event.symbol)
      order_type = event.order_type
      quantity = event.quantity
      direction = event.direction
      params = event.params

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
        print("ORDER FILLED")
        self.events.put(fill)
      else:
        print("ORDER COULD NOT BE FILLED")

    def execute_take_profit(self, event):
      print("Executing take profit order")
      exchange = event.exchange
      symbol = from_standard_to_exchange_notation(exchange, event.symbol)
      order_type = event.order_type
      quantity = event.quantity
      direction = event.direction
      params = event.params

      order = self.exchanges[event.exchange].create_order(symbol, order_type, direction, quantity, None, params)
      print('TAKE PROFIT ORDER SET')

    def execute_stop_loss(self, event):
      print("Executing stop loss order")
      exchange = event.exchange
      symbol = from_standard_to_exchange_notation(exchange, event.symbol)
      order_type = event.order_type
      quantity = event.quantity
      direction = event.direction
      params = event.params

      order = self.exchanges[event.exchange].create_order(symbol, order_type, direction, quantity, None, params)
      print('STOP LOSS ORDER SET')

    def execute_close_position(self, event):
      print('Executing close {} position'.format(event.symbol))
      exchange = event.exchange
      symbol = from_standard_to_exchange_notation(exchange, event.symbol)
      params = { 'symbol': symbol }

      self.exchanges[exchange].private_post_order_closeposition(params)
      print('CLOSE POSITION DONE')

    def execute_cancel_all_orders(self, event):
      print('Executing close {} position'.format(event.symbol))
      exchange = event.exchange
      symbol = from_standard_to_exchange_notation(exchange, event.symbol)
      params = { 'symbol': symbol }

      self.exchanges[exchange].private_delete_order_all(params)
      print('CANCEL ALL ORDERS')


    def execute_order(self, event):
      """
      Parameters:
      event - Contains an Event object with order information.
      """
      if event.type == 'ORDER':
        if event.order_type == "Market":
          self.execute_market_order(event)
        elif event.order_type == "MarketIfTouched":
          self.execute_take_profit(event)
        elif event.order_type == "Stop":
          self.execute_stop_loss(event)
        elif event.order_type == "Limit":
          self.execute_limit_order(event)
        elif event.order_type == "CancelAll":
          self.execute_cancel_all_orders(event)
        elif event.order_type == "ClosePosition":
          self.execute_close_position(event)

      time.sleep(1)
      self.order_id += 1










    # def execute_orders(self, event):
    #   if event.type == "BULK_ORDER":
    #     self.execute_bulk_order(event)



    # def execute_bulk_order(self, event):
    #   events = event.events
    #   orders = []

    #   for e in events:
    #     # Adapt direction
    #     direction = { 'sell': 'Sell', 'buy': 'Buy' }[e.direction]
    #     order = {
    #       "symbol": from_standard_to_exchange_notation(e.exchange, e.symbol),
    #       "side": direction,
    #       "ordType": e.order_type,
    #       "orderQty": e.quantity or 1
    #     }

    #     if 'stopPx' in e.params:
    #       order['stopPx'] = e.params['stopPx']

    #     if 'execInst' in e.params:
    #       order['execInst'] = e.params['execInst']

    #     orders.append(order)


    #   order_payload = json.dumps(orders)
    #   payload = { 'orders': order_payload }
    #   print(payload)
    #   result = self.exchanges['bitmex'].private_post_order_bulk(payload)
    #   print(result)




# try to call a unified method
# try:
#     response = await exchange.fetch_order_book('ETH/BTC')
#     print(ticker)
# except ccxt.NetworkError as e:
#     print(exchange.id, 'fetch_order_book failed due to a network error:', str(e))
#     # retry or whatever
#     # ...
# except ccxt.ExchangeError as e:
#     print(exchange.id, 'fetch_order_book failed due to exchange error:', str(e))
#     # retry or whatever
#     # ...
# except Exception as e:
#     print(exchange.id, 'fetch_order_book failed with:', str(e))
#     # retry or whatever
#     # ...