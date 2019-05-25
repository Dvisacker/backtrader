from __future__ import print_function

import time

from .execution import ExecutionHandler
from abc import ABCMeta, abstractmethod
from event import FillEvent, OrderEvent
from datetime import datetime

class SimulatedCryptoExchangeExecutionHandler(ExecutionHandler):
    """
    The simulated execution handler simply converts all order
    objects into their equivalent fill objects automatically
    without latency, slippage or fill-ratio issues.
    This allows a straightforward "first go" test of any strategy,
    before implementation with a more sophisticated execution
    handler.
    """

    def __init__(self, events, configuration):
        """
        Initialises the handler, setting the event queues
        up internally.
        :param events: The Queue of Event objects.
        """
        self.events = events
        self.exchange = configuration.exchange_names[0]

    def execute_order(self, event):
        """
        Simply converts Order objects into Fill objects naively,
        i.e. without any latency, slippage or fill ratio problems.
        :param event: Contains an Event object with order information.
        """
        if event.type == 'ORDER':
            fill_event = FillEvent(
                datetime.utcnow(),
                event.symbol,
                self.exchange,
                event.quantity,
                event.direction,
                None
            )

            self.events.put(fill_event)