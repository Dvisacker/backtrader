#!/usr/bin/python
# -*- coding: utf-8 -*-

# event.py

from __future__ import print_function


class Event(object):
    """
    Event is base class providing an interface for all subsequent
    (inherited) events, that will trigger further events in the
    trading infrastructure.
    """
    pass


class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with
    corresponding bars.
    """

    def __init__(self, data=None, timestamp=None):
        """
        Initialises the MarketEvent.
        """
        self.type = 'MARKET'
        self.data = data
        self.timestamp = timestamp

    def print_market_event(self):
      """
      Outputs tick data
      """
      print(
          "MARKET: Data=%s, Timestamp=%s" %
          (self.data, self.timestamp)
      )


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """

    def __init__(self, strategy_id, exchange, symbol, datetime, signal_type, strength):
        """
        Initialises the SignalEvent.
        Parameters:
        strategy_id - The unique ID of the strategy sending the signal.
        symbol - The ticker symbol, e.g. 'GOOG'.
        datetime - The timestamp at which the signal was generated.
        signal_type - 'LONG' or 'SHORT'.
        strength - An adjustment factor "suggestion" used to scale quantity at the portfolio level. Useful for pairs strategies.
        """
        self.strategy_id = strategy_id
        self.exchange = exchange
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength

    def print_signal(self):
        """
        Outputs the values within the Order.
        """
        print(
            "SIGNAL: Exchange=%s, Symbol=%s, Type=%s, Strength=%s" %
            (self.exchange, self.symbol, self.signal_type, self.strength)
        )

class SignalEvents(object):
      def __init__(self, events, id):
        self.type = 'SIGNAL'
        self.id = id
        self.events = events

      def print_signals(self):
        """
        Outputs the values of all signals within the SignalEvents object
        """
        for e in self.events:
          e.print_signal()


class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (e.g. GOOG), a type (market or limit),
    quantity and a direction.
    """

    def __init__(self, exchange, symbol, order_type, quantity=0, direction='', leverage=1, price=None, params={}):
        """
        Initialises the order type, setting whether it is
        a Market order ('MKT') or Limit order ('LMT'), has
        a quantity (integral) and its direction ('BUY' or
        'SELL').
        TODO: Must handle error checking here to obtain
        rational orders (i.e. no negative quantities etc).
        Parameters:
        symbol - The instrument to trade.
        order_type - 'MKT' or 'LMT' for Market or Limit.
        quantity - Non-negative integer for quantity.
        direction - 'BUY' or 'SELL' for long or short.
        """
        self.type = 'ORDER'
        self.exchange = exchange
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.leverage = leverage
        self.price = price
        self.params = params

    def print_order(self):
        """
        Outputs the values within the order
        """
        print(
            "ORDER: Exchange=%s, Symbol=%s, Type=%s, Quantity=%s, Direction=%s, Price=%s" %
            (self.exchange, self.symbol, self.order_type, self.quantity, self.direction, self.price)
        )

class BulkOrderEvent(object):
      def __init__(self, events):
        self.type = 'BULK_ORDER'
        self.events = events


class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the fee of the trade from the brokerage.

    TODO: Currently does not support filling positions at
    different prices. This will be simulated by averaging
    the cost.
    """

    def __init__(self, timeindex, symbol, exchange, quantity,
                 direction, price=None, fee=None, leverage=None, entry_price=None,
                 fill_type=None):
        """
        Initialises the FillEvent object. Sets the symbol, exchange,
        quantity, direction, cost of fill and an optional
        fee.
        If fee is not provided, the Fill object will
        calculate it based on the trade size and Interactive
        Brokers fees.
        Parameters:
        timeindex - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        exchange - The exchange where the order was filled.
        quantity - The filled quantity.
        direction - The direction of fill ('BUY' or 'SELL')
        fill_cost - The holdings value in dollars.
        fee - An optional fee
        """
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.price = price
        self.leverage = leverage

        if entry_price is None:
          self.entry_price = 0
        else:
          self.entry_price = entry_price

        # Compute fee
        if fee is None:
            self.fee = self.compute_fees()
        else:
            self.fee = fee

        if fill_type is None:
            self.fill_type = 'Market'
        else:
            self.fill_type = fill_type

    def print_fill(self):
        """
        Outputs the values within the Order.
        """
        print(
            "FILL: Exchange=%s, Symbol=%s, Quantity=%s, Direction=%s, Price=%s, Type=%s" %
            (self.exchange, self.symbol, self.quantity, self.direction, self.price, self.fill_type)
        )

    def compute_fees(self):
        """
        The default bitmex fee is 0.0750% which I left as default for now
        """
        fee = 0.0750 / 100
        return fee


