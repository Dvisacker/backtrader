# '''
# Copyright (C) 2017-2019  Bryant Moscon - bmoscon@gmail.com
# Please see the LICENSE file for the terms and conditions
# associated with this software.
# '''
# from cryptofeed.callback import TickerCallback, TradeCallback, BookCallback, FundingCallback
# from cryptofeed import FeedHandler
# from cryptofeed.exchanges import Bitmex, Coinbase, Bitfinex, Poloniex, Gemini, HitBTC, Bitstamp, Kraken, Binance, EXX, Huobi, HuobiUS, OKCoin, OKEx, Coinbene
# from cryptofeed.defines import L3_BOOK, L2_BOOK, BID, ASK, TRADES, TICKER, FUNDING, COINBASE


# # Examples of some handlers for different updates. These currently don't do much.
# # Handlers should conform to the patterns/signatures in callback.py
# # Handlers can be normal methods/functions or async. The feedhandler is paused
# # while the callbacks are being handled (unless they in turn await other functions or I/O)
# # so they should be as lightweight as possible
# async def ticker(feed, pair, bid, ask):
#     print(f'Feed: {feed} Pair: {pair} Bid: {bid} Ask: {ask}')


# async def trade(feed, pair, order_id, timestamp, side, amount, price):
#     print(f"Timestamp: {timestamp} Feed: {feed} Pair: {pair} ID: {order_id} Side: {side} Amount: {amount} Price: {price}")


# async def book(feed, pair, book, timestamp):
#     print(f'Timestamp: {timestamp} Feed: {feed} Pair: {pair} Book Bid Size is {len(book[BID])} Ask Size is {len(book[ASK])}')


# async def funding(**kwargs):
#     print(f"Funding Update for {kwargs['feed']}")
#     print(kwargs)


# def main():
#     f = FeedHandler()
#     f.add_feed(Bitmex(pairs=['XBTUSD'], channels=[L3_BOOK], callbacks={L3_BOOK: BookCallback(book)}))
#     f.run()


# if __name__ == '__main__':
#     main()





import logging
import os

from time import sleep
from apis import BitMEXWebsocket

# Basic use of websocket.
def run():
    logger = setup_logger()

    # Instantiating the WS will make it connect. Be sure to add your api_key/api_secret.
    ws = BitMEXWebsocket(endpoint="https://testnet.bitmex.com/api/v1", symbol="XBTUSD",
    api_key=os.environ['BITMEX_TEST_KEY_ID'], api_secret=os.environ['BITMEX_TEST_KEY_SECRET'])

    # logger.info("Instrument data: %s" % ws.get_instrument())
    # logger.info("Positions: %s" % ws.positions())

    # Run forever
    while(ws.ws.sock.connected):
      if ws.api_key:
        # logger.info("Funds: %s" % ws.funds())
        logger.info("Positions: %s" % ws.positions('XBTUSD'))

        # logger.info("Ticker: %s" % ws.get_ticker())
        # if ws.api_key:
        #     logger.info("Funds: %s" % ws.funds())
        # logger.info("Market Depth: %s" % ws.market_depth())
        # logger.info("Recent Trades: %s\n\n" % ws.recent_trades())
        sleep(10)


def setup_logger():
    # Prints logger info to terminal
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Change this to DEBUG if you want a lot more info
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    run()
