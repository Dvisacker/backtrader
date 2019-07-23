import os
import pymongo
import pandas as pd

from datetime import datetime
from utils.log import logger

class CSVHandler(object):

  def __init__(self, csv_file):
    self.csv_file = csv_file

  def insert_portfolio(self, portfolio):
    try:
      with open(self.csv_file, "w") as f:
        writer = csv.DictWriter(a, )
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