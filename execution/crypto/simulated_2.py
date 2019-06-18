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

class SimulatedCryptoExchangeExecutionHandler(ExecutionHandler):
    """
    Handles order execution via the Interactive Brokers
    API, for use against accounts when trading live
    directly.
    """

    def __init__(self, events, configuration):
        """
        Initialises the BitmexExecutionHandler instance.
        Parameters:
        events - The Queue of Event objects.
        """
        self.events = events
        self.take_profit_gap = 0.5
        self.stop_loss_gap = 0.5
        self.fill_dict = {}
        self.order_id = 1

    def execute_market_order(self, event):
      exchange = event.exchange
      symbol = event.symbol
      quantity = event.quantity
      direction = event.direction

      # Create a fill event object
      fill = FillEvent(
          datetime.datetime.utcnow(), symbol,
          exchange, quantity, direction,
      )

      # Place the fill event onto the event queue
      self.events.put(fill)

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
        elif event.order_type == "ClosePosition":
          self.execute_close_position(event)

      self.order_id += 1









