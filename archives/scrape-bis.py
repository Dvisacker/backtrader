import numpy as np
import datetime
import time
import re
import pandas as pd
import requests
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exchange')
parser.add_argument('-p','--pair')
args = parser.parse_args()
 
pair = args.pair
exchange = args.exchange
 
resp = requests.get("https://api.cryptowat.ch/markets/{}/{}/ohlc?after=1000000000&periods=86400".format(exchange, pair))
resp = resp.json()
resp = resp["result"]["86400"]
 
all_candles = []
for c in resp:
    all_candles.append({"Date": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(c[0])), "Open": float(c[1]), "High": float(c[2]), "Low": float(c[3]), "Close": float(c[4]), "Volume": float(c[5])})
 
df = pd.DataFrame(all_candles)
df = df.set_index(["Date"])
df = df[["Open", "High", "Low", "Close", "Volume"]]
print df
df.to_csv("{}_{}_1440.csv".format(exchange, pair))