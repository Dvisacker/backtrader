import pymongo

class MongoWriter(object):

  def __init__(self):
    self.db_client = pymongo.MongoClient('localhost', 27017)
    self.db = self.db_client.bot_db

    self.db.positions.create_index([('datetime', pymongo.ASCENDING)], unique=True)
    self.db.holdings.create_index([('datetime', pymongo.ASCENDING)], unique=True)

  def insert_positions(self, positions):
    self.db.positions.insert_one(positions)

  def insert_holdings(self, holdings):
    self.db.holdings.insert_one(holdings)