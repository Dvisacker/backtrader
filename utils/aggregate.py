import math
from cryptofeed.callback import Callback
from cryptofeed.backends.aggregate import AggregateCallback
from cryptofeed import FeedHandler
from cryptofeed.exchanges import Bitmex
from cryptofeed.defines import TRADES
from datetime import datetime, timedelta
from decimal import Decimal
from .helpers import from_standard_to_exchange_notation
from threading import Lock

import pdb


def ceil_dt(dt, delta):
    return datetime.min + math.ceil((dt - datetime.min) / delta) * delta


class OHLCV(AggregateCallback):
    """
    Aggregate trades and calculate OHLCV for time window
    window is in seconds, defaults to 300 seconds (5 minutes)
    """

    def __init__(self, *args, start_time, exchange, instruments, window=300, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window
        self.exchange = exchange
        self.start_time = datetime.fromtimestamp(start_time)
        self.last_update = datetime.fromtimestamp(start_time)
        self.instruments = [from_standard_to_exchange_notation(exchange, i) for i in instruments[exchange]]
        self.previous_data = {}
        self.data = {}
        # self.mutex = Lock()

    def _update(self, pair, amount, price):
        if pair not in self.data:
            self.data[pair] = {'open': price, 'high': price, 'low': price, 'close': price, 'volume': 0, 'vwap': 0}

        self.data[pair]['close'] = price
        self.data[pair]['volume'] += amount

        if price > self.data[pair]['high']:
            self.data[pair]['high'] = price
        if price < self.data[pair]['low']:
            self.data[pair]['low'] = price
        self.data[pair]['vwap'] += price * amount

    async def __call__(self, *, feed: str, pair: str, side: str, amount: Decimal, price: Decimal, order_id=None, timestamp=None):
        now = datetime.now()
        amount = float(amount)
        price = float(price)

        if now < self.start_time:
            return

        if now - self.last_update > timedelta(seconds=self.window):
            self.last_update = self.last_update + timedelta(seconds=self.window)
            for i in self.instruments:
            # Case where the OHLCV for this instrument is not initialized
              if i not in self.data and i not in self.previous_data:
                  self.data[i] = {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0, 'vwap': 0}
                  self.data[i]['timestamp'] = self.last_update
                  self.data[i]['time'] = datetime.timestamp(self.last_update)

              # We add fields for pairs that are not updated.
              elif i not in self.data:
                  self.data[i] = self.previous_data[i]
                  self.data[i]['volume'] = 0.0
                  self.data[i]['vwap'] = 0.0
                  self.data[i]['timestamp'] = self.last_update
                  self.data[i]['time'] = datetime.timestamp(self.last_update)

              # Normal Case
              else:
                  self.data[i]['vwap'] /= self.data[i]['volume']
                  self.data[i]['timestamp'] = self.last_update
                  self.data[i]['time'] = datetime.timestamp(self.last_update)

            await self.handler(data=self.data, timestamp=self.last_update)
            self.previous_data = self.data
            self.data = {}

        else:
            self._update(pair, amount, price)
