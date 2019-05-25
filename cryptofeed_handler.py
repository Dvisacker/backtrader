import math

from cryptofeed.callback import Callback
from cryptofeed.backends.aggregate import AggregateCallback
from cryptofeed import FeedHandler
from cryptofeed.exchanges import Bitmex
from cryptofeed.defines import TRADES
from utils.aggregate import OHLCV
from datetime import datetime, timedelta
from decimal import Decimal

# def ceil_dt(dt, delta):
#     return datetime.min + math.ceil((dt - datetime.min) / delta) * delta

# class OHLCV(AggregateCallback):
#     """
#     Aggregate trades and calculate OHLCV for time window
#     window is in seconds, defaults to 300 seconds (5 minutes)
#     """

#     def __init__(self, *args, window=300, **kwargs):
#         super().__init__(*args, **kwargs)

#         now = datetime.now()
#         delta = timedelta(seconds=window)
#         self.window = window
#         self.start_time = datetime.min + math.ceil((now - datetime.min) / delta) * delta
#         self.last_update = self.start_time
#         self.data = {}

#     def _agg(self, pair, amount, price):
#         if pair not in self.data:
#             self.data[pair] = {'open': price, 'high': price, 'low': price, 'close': price, 'volume': Decimal(0), 'vwap': Decimal(0)}

#         self.data[pair]['close'] = price
#         self.data[pair]['volume'] += amount
#         if price > self.data[pair]['high']:
#             self.data[pair]['high'] = price
#         if price < self.data[pair]['low']:
#             self.data[pair]['low'] = price
#         self.data[pair]['vwap'] += price * amount

#     async def __call__(self, *, feed: str, pair: str, side: str, amount: Decimal, price: Decimal, order_id=None, timestamp=None):
#         now = datetime.now()

#         if now < self.start_time:
#           return

#         if now - self.last_update > timedelta(seconds=self.window):
#             self.last_update = self.last_update + timedelta(seconds=self.window)
#             for p in self.data:
#                 self.data[p]['vwap'] /= self.data[p]['volume']
#                 self.data[p]['timestamp'] = self.last_update
#                 self.data[p]['time'] = datetime.timestamp(self.last_update)
#             await self.handler(data=self.data)
#             self.data = {}

#         self._agg(pair, amount, price)

async def ohlcv(data=None):
    print(data)

def main():
    f = FeedHandler()
    f.add_feed(Bitmex(pairs=['XBTUSD', 'ETHUSD'], channels=[TRADES], callbacks={TRADES: OHLCV(Callback(ohlcv), window=10)}))

    f.run()


if __name__ == '__main__':
    main()