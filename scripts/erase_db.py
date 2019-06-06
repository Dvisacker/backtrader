import os
from datetime import datetime

if __name__ == "__main__" and __package__ is None:
  from sys import path
  from os.path import dirname as dir
  path.append(dir(path[0]))
  __package__ = "scripts"

from db.mongo_handler import MongoHandler

mongo_handler = MongoHandler()
mongo_handler.erase()
