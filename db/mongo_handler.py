import os
import pymongo
import pandas as pd

from datetime import datetime
from utils.log import logger

class MongoHandler(object):

  def __init__(self):
    self.db_client = pymongo.MongoClient('db', 27017)
    self.db = self.db_client.bot_db

    self.db.portfolios.create_index([('datetime', pymongo.ASCENDING)], unique=True)

  def insert_portfolio(self, portfolio):
    try:
      self.db.portfolios.insert_one(portfolio)
    except Exception as e:
      logger.error('Could not save current position: {}'.format(e))

  def read_portfolios(self):
    cursor = self.db.portfolios.find({})
    df = pd.DataFrame(list(cursor))

    if '_id' in df:
      del df['_id']

    return df

  def erase(self):
    self.db_client.drop_database('bot_db')

  # def dump_to_csv(self):
  #   positions = self.read_positions()
  #   holdings = self.read_holdings()

  #   positions_indexed = positions.set_index(['datetime'])
  #   holdings_indexed = holdings.set_index(['datetime'])
  #   all_data = pd.concat([positions_indexed, holdings_indexed], axis=1, sort=False)

  #   folder_name = './saved/dump-{}'.format(datetime.utcnow())
  #   positions_filename = os.path.join(folder_name, 'positions.csv')
  #   holdings_filename = os.path.join(folder_name, 'holdings.csv')
  #   all_data_filename = os.path.join(folder_name, 'data.csv')

  #   try:
  #     os.mkdir(folder_name)
  #   except OSError:
  #     logger.error("Creation of the directory failed %s" % folder_name)
  #   else:
  #     all_data.to_csv(all_data_filename)
  #     positions_indexed.to_csv(positions_filename)
  #     holdings_indexed.to_csv(holdings_filename)







