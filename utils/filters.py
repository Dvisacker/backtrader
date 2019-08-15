import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_notebook

def compute_cusum_return_events(close, h):
  """
  This CUSUM filter computes potential buy/sell signals if an
  absolute return h is observed relative to a prior high or low
  """
  time_events, s_pos, s_neg = [], 0, 0
  diff = np.log(close).diff().dropna()
  for i in tqdm(diff.index[1:]):
      pos, neg = float(s_pos+diff.loc[i]), float(s_neg+diff.loc[i])
      s_pos, s_neg = max(0., pos), min(0., neg)

      if s_neg<-h:
          s_neg=0
          time_events.append(i)
      elif s_pos>h:
          s_pos=0
          time_events.append(i)

  return pd.DatetimeIndex(time_events)
