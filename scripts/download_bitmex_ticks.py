#!/usr/bin/env python3
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "scripts"


import argparse
from utils.scrape import scrape_bitmex_trades

parser = argparse.ArgumentParser(description='Market data downloader')
parser.add_argument('-from', '--from_date',
                      type=str,
                      help='The date from which to start dowloading ohlcv from'
                    )

parser.add_argument('-to', '--to_date',
                      type=str,
                      help='The date up to which to download ohlcv to'
                    )

args = parser.parse_args()

scrape_bitmex_trades(args.from_date, args.to_date)


