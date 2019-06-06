import os
import pymongo
import pandas as pd

from datetime import datetime

class MongoHandler(object):

  def __init__(self):
    self.db_client = pymongo.MongoClient('localhost', 27017)
    self.db = self.db_client.bot_db

    self.db.positions.create_index([('datetime', pymongo.ASCENDING)], unique=True)
    self.db.holdings.create_index([('datetime', pymongo.ASCENDING)], unique=True)

  def insert_positions(self, positions):
    try:
      self.db.positions.insert_one(positions)
    except Exception as e:
      print('Could not save current position: {}'.format(e))

  def insert_holdings(self, holdings):
    try:
      self.db.holdings.insert_one(holdings)
    except Exception as e:
      print('Could not save current holdings: {}'.format(e))

  def read_positions(self):
    cursor = self.db.positions.find({})
    df = pd.DataFrame(list(cursor))

    if '_id' in df:
      del df['_id']

    return df

  def read_holdings(self):
    cursor = self.db.holdings.find({})
    df = pd.DataFrame(list(cursor))

    if '_id' in df:
      del df['_id']

    return df

  def erase(self):
    self.db_client.drop_database('bot_db')

  def dump_to_csv(self):
    positions = self.read_positions()
    holdings = self.read_holdings()

    positions_indexed = positions.set_index(['datetime'])
    holdings_indexed = holdings.set_index(['datetime'])
    all_data = pd.concat([positions_indexed, holdings_indexed], axis=1, sort=False)

    folder_name = './saved/dump-{}'.format(datetime.now())
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







