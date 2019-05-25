import ccxt
import os

def create_exchange_instances(exchange_names):
  exchanges = {}

  for e in exchange_names:
    if e == 'bitmex':
      bitmex = ccxt.bitmex({
        'apiKey': os.environ['BITMEX_TEST_KEY_ID'],
        'secret': os.environ['BITMEX_TEST_KEY_SECRET'],
        'enableRateLimit': True
      })

      if 'test' in bitmex.urls:
        bitmex.urls['api'] = bitmex.urls['test']

      exchanges['bitmex'] = bitmex

  return exchanges