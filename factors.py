import pandas as pd
import numpy as np

def momentum(data, timeperiod):
  return data[-timeperiod] / data[0]


def volatility(data):
  df = pd.DataFrame(data=data)
  return 1 / np.log(data).diff().std()


def average_volume_traded(prices, volumes):
  return np.mean(prices * volumes, axis = 0)