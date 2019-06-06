import os
import pandas as pd
from datetime import datetime

if __name__ == "__main__" and __package__ is None:
  from sys import path
  from os.path import dirname as dir
  path.append(dir(path[0]))
  __package__ = "scripts"

from db.mongo_handler import MongoHandler

mongo_handler = MongoHandler()
positions = mongo_handler.read_positions()
holdings = mongo_handler.read_holdings()
positions_indexed = positions.set_index(['datetime'])
holdings_indexed = holdings.set_index(['datetime'])
all_data = pd.concat([positions_indexed, holdings_indexed], axis=1, sort=False)

folder_name = './saved/dump-{}'.format(datetime.utcnow())
positions_filename = os.path.join(folder_name, 'positions.csv')
holdings_filename = os.path.join(folder_name, 'holdings.csv')
all_data_filename = os.path.join(folder_name, 'data.csv')

try:
  os.mkdir(folder_name)
except OSError:
  print("Creation of the directory failed %s" % folder_name)
else:
  all_data.to_csv(all_data_filename)
  positions_indexed.to_csv(positions_filename)
  holdings_indexed.to_csv(holdings_filename)




