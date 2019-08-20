import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb

from sklearn.model_selection import cross_val_score
from sklearn import metrics


def add_barriers_on_timestamps(close, volatility):
  """
  This function adds barriers on an unfiltered series of prices
  close: indexed series of close prices
  events: index representing potential entry signals (for example computed by a cusum filter)
  volatility: 30min volatility of the time series
  """
  #1) compute vertical barriers
  events = close.index
  side = pd.Series(1, index=events)
  vertical_barriers=close.index.searchsorted(events+pd.Timedelta(minutes=30))
  vertical_barriers=vertical_barriers[vertical_barriers<close.shape[0]]
  vertical_barriers=(pd.Series(close.index[vertical_barriers],index=events[:vertical_barriers.shape[0]]))

  #2) compute thresholds
  threshold=volatility.loc[events]

  #2) form events object, apply stop loss on the vertical barriers
  # side=pd.Series(1.,index=threshold.index)
  events=pd.concat({'vertical_barrier': vertical_barriers, 'threshold': threshold,'side': side }, axis=1).dropna(subset=['threshold'])
  df0 = apply_profit_taking_and_stop_losses(close, events)

  #3) we replace NaT by a very large date in order to compare values
  df0['take_profit'][pd.isnull(df0['take_profit'])] = pd.Timestamp(pd.Timestamp(2100, 1, 1, 1), tz='US/Eastern')
  df0['stop_loss'][pd.isnull(df0['stop_loss'])] = pd.Timestamp(pd.Timestamp(2100, 1, 1, 1), tz='US/Eastern')

  events['entry_signal'] = df0.index
  events['take_profit'] = df0['take_profit']
  events['stop_loss'] = df0['stop_loss']
  events['vertical_barrier']=df0['vertical_barrier']
  events['exit_signal'] = df0.dropna(how='all').min(axis=1)

  return events

def add_barriers_on_trade_signals(close,events,volatility):
  """
  This function adds barriers on an unfiltered series of prices
  close: indexed series of close prices
  events: index representing potential entry signals (for example computed by a cusum filter)
  volatility: 30min volatility of the time series
  """
  #1) compute vertical barriers
  side = pd.Series(1., index=events)
  vertical_barriers=close.index.searchsorted(events+pd.Timedelta(minutes=30))
  vertical_barriers=vertical_barriers[vertical_barriers<close.shape[0]]
  vertical_barriers=(pd.Series(close.index[vertical_barriers],index=events[:vertical_barriers.shape[0]]))


  #2) compute thresholds
  threshold=volatility.loc[events]

  #2) form events object, apply stop loss on the vertical barriers
  # side=pd.Series(1.,index=threshold.index)
  events=pd.concat({'vertical_barrier': vertical_barriers, 'threshold': threshold,'side': side }, axis=1).dropna(subset=['threshold'])
  df0 = apply_profit_taking_and_stop_losses(close, events)

  #3) we replace NaT by a very large date in order to compare values
  df0['take_profit'][pd.isnull(df0['take_profit'])] = pd.Timestamp(pd.Timestamp(2100, 1, 1, 1), tz='US/Eastern')
  df0['stop_loss'][pd.isnull(df0['stop_loss'])] = pd.Timestamp(pd.Timestamp(2100, 1, 1, 1), tz='US/Eastern')

  events['entry_signal'] = df0.index
  events['take_profit'] = df0['take_profit']
  events['stop_loss'] = df0['stop_loss']
  events['vertical_barrier']=df0['vertical_barrier']
  events['exit_signal'] = df0.dropna(how='all').min(axis=1)

  return events


def add_barriers_on_buy_sell_signals(close, side, stop_thresholds):
    """
    This function adds barriers on a series of events determined by an algorithm.
    Therefore the requirement arguments are
    close: indexed series of close prices
    events: index entry signals determined by an algorithm
    side: the side of the events
    """
    #1) compute vertical barriers
    events = side.index
    vertical_barriers = pd.Series(events + pd.Timedelta(minutes=30), index=events)
    # vertical_barriers = events + pd.Timedelta(minutes=30)
    # vertical_barriers=events.searchsorted(events+pd.Timedelta(minutes=30))
    # vertical_barriers=vertical_barriers[vertical_barriers<close.shape[0]]
    # vertical_barriers=(pd.Series(close.index[vertical_barriers],index=events[:vertical_barriers.shape[0]]))
    # pdb.set_trace()

    # pdb.set_trace()

    #2) compute thresholds
    threshold=stop_thresholds.loc[events]

    #2) form events object, apply stop loss on the vertical barriers
    # side=pd.Series(1.,index=threshold.index)
    events=pd.concat({'vertical_barrier': vertical_barriers, 'threshold': threshold,'side': side }, axis=1).dropna(subset=['threshold'])
    df0 = apply_profit_taking_and_stop_losses(close, events)

    #3) we replace NaT by a very large date in order to compare values
    df0['take_profit'][pd.isnull(df0['take_profit'])] = pd.Timestamp(2100, 1, 1, 1)
    df0['stop_loss'][pd.isnull(df0['stop_loss'])] = pd.Timestamp(2100, 1, 1, 1)

    events['entry_signal'] = df0.index
    events['take_profit'] = df0['take_profit']
    events['stop_loss'] = df0['stop_loss']
    events['vertical_barrier']=df0['vertical_barrier']

    events['exit_signal'] = df0.dropna(how='all').min(axis=1)
    return events


def apply_profit_taking_and_stop_losses(close,events):
  out = events[['vertical_barrier']].copy(deep=True)
  take_profits=events['threshold']
  stop_losses=-events['threshold']

  # Close has a different index fuck this shit
  for start,end in events['vertical_barrier'].iteritems():
      df0=close[start:end] # path prices
      df0=(df0/close[start]-1)*events.at[start,'side'] # path returns
      out.loc[start,'stop_loss']=df0[df0<stop_losses[start]].index.min() # earliest stop loss
      out.loc[start,'take_profit']=df0[df0>take_profits[start]].index.min() # earliest profit taking

  return out


def compute_labels(events, close):
    '''
    In this algorithm, the trading side have been chosen so instead of having {1, -1} labels,
    we have labels that are either 0 or 1. A 0 label means we decide not too trade,
    a 1 label means we decide to trade.
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's start time
    -events['vertical_barrier'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) compute prices at time of exit with the backfill method
    events = events.copy()
    events = events.dropna(subset=['exit_signal'])
    price_index = events.index.union(events['exit_signal']).drop_duplicates()
    prices = close.reindex(price_index, method='bfill')

    #2) create labels object
    labels = pd.DataFrame(index=events.index)
    entry_signals = events.index
    exit_signals = events['exit_signal']
    labels['returns'] = prices.loc[exit_signals].values / prices.loc[entry_signals] - 1

    #3) meta-labeling
    labels['returns'] *= events['side']
    labels['label'] = np.sign(labels['returns'])
    labels.loc[labels['returns'] <= 0,'label'] = 0

    return labels


def add_labels(events, close):
    events = events.copy()
    labels = compute_labels(events, close)
    events = pd.concat([events, labels], axis=1)

    return events

def get_daily_volatility(close,time_period=60):
  return close.ewm(time_period).std()

def compute_classification_params(close):
  volatility = get_daily_volatility(close)
  mean_volatility = volatility.mean()
  horizontal_barrier_threshold = 2000 * mean_volatility
  minimum_return = 1.5 * mean_volatility

  return {
    'volatility': volatility,
    'mean_volatility': mean_volatility,
    'horizontal_barrier_threshold': horizontal_barrier_threshold,
    'minimum_return': minimum_return
  }


def plot_roc_curve(fpr, tpr):
  plt.figure()
  plt.plot([0,1],[1,0])
  plt.plot(fpr, tpr, label='RF')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC Curve')
  plt.legend(loc='best')
  plt.show()

def print_regressor_metrics(model, X,y, kfold=None, scores=None):
  print('Scores:', scores)
  print('Mean score: ', np.mean(scores))

  scoring = 'neg_mean_absolute_error'
  results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
  print("MAE: {} ({})".format(results.mean(), results.std()))

  scoring = 'neg_mean_squared_error'
  results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
  print("MSE: {} ({})".format(results.mean(), results.std()))

  scoring = 'r2'
  results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
  print("R^2: {} ({})".format(results.mean(), results.std()))


def print_reports(y_test, y_test_pred):
  print(metrics.classification_report(y_test, y_test_pred, target_names=['short', 'no_trade', 'long']))
  print(pd.crosstab(y_test, y_test_pred, rownames=['Actual labels'], colnames=['Predicted labels']))

def residuals_plot(model, X_train, y_train, X_test, y_test):
  visualizer = ResidualsPlot(model)
  visualizer.fit(X_train, y_train)
  visualizer.score(X_test, y_test)
  visualizer.poof()

def prediction_error_plot(model, X_train, y_train, X_test, y_test):
  visualizer = PredictionError(model)
  visualizer.fit(X_train, y_train)
  visualizer.score(X_test, y_test)
  visualizer.poof()

def create_confusion_matrix(y_pred, y_true):
  df = pd.DataFrame({'pred': y_pred, 'true': y_true })
  df['pred_sign'] = np.sign(df['pred'])
  df['true_sign'] = np.sign(df['true'])
  df['correct'] = df['pred_sign'] == df['true_sign']

  df2 = df[df['true_sign'] != 0]
  print(df2['correct'].value_counts())



  # def add_barriers_advanced(close,events,side,volatility,minimum_return):
  #   """
  #   close: indexed series of close prices
  #   events: index representing potential entry signals (for example computed by a cusum filter)
  #   volatility: 30min volatility of the time series
  #   """
  #   #1) compute vertical barriers
  #   vertical_barriers=close.index.searchsorted(events+pd.Timedelta(minutes=30))
  #   vertical_barriers=vertical_barriers[vertical_barriers<close.shape[0]]
  #   vertical_barriers=(pd.Series(close.index[vertical_barriers],index=events[:vertical_barriers.shape[0]]))

  #   #2) compute thresholds
  #   threshold=volatility.loc[events]
  #   threshold=threshold[threshold>minimum_return]
  #   # This probably leaves NAN for rows where the threshold is not exceeeded

  #   #2) form events object, apply stop loss on the vertical barriers
  #   # side=pd.Series(1.,index=threshold.index)
  #   events=pd.concat({'vertical_barrier': vertical_barriers, 'threshold': threshold,'side': side }, axis=1).dropna(subset=['threshold'])
  #   df0 = apply_profit_taking_and_stop_losses(close, events)

  #   #3) we replace NaT by a very large date in order to compare values
  #   df0['take_profit'][pd.isnull(df0['take_profit'])] = pd.Timestamp(pd.Timestamp(2100, 1, 1, 1), tz='US/Eastern')
  #   df0['stop_loss'][pd.isnull(df0['stop_loss'])] = pd.Timestamp(pd.Timestamp(2100, 1, 1, 1), tz='US/Eastern')

  #   events['entry_signal'] = df0.index
  #   events['take_profit'] = df0['take_profit']
  #   events['stop_loss'] = df0['stop_loss']
  #   events['vertical_barrier']=df0['vertical_barrier']
  #   events['exit_signal'] = df0.dropna(how='all').min(axis=1)

  #   return events


# def compute_labels_4(events, close):
#   """
#   Compute labels without metalabels (e.g. sides determined by a previous algorithm)
#   """
#   #1) prices aligned with events
#   events = events.copy()
#   events = events.dropna(subset=['vertical_barriers'])
#   price_index = events.index.union(events['vertical_barriers']).drop_duplicates()
#   prices = close.reindex(price_index, method='bfill')

#   #2) create return object
#   labels = pd.DataFrame(index=events.index)
#   entry_signals = events.index
#   exit_signals = events['exit_signal']
#   labels['returns'] = prices.loc[exit_signals].values / prices.loc[entry_signals] - 1

#   vtouch_first_idx = events[events['exit_signal'].isin(events['vertical_barrier'])].index
#   labels.loc[vtouch_first_idx, 'label'] = 0.

#   return labels

# def compute_labels(events, close):
#   '''
#   Compute event's outcome (including side information, if provided).
#   events is a DataFrame where:
#   -events.index is event's start time
#   -events['vertical_barrier'] is event's endtime
#   -events['trgt'] is event's target
#   -events['side'] (optional) implies the algo's position side
#   Case 1: ('side' not in events): bin in (-1,1) <-label by price action
#   Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
#   '''
#   #1) compute prices at time of exit with the backfill method
#   events = events.copy()
#   events = events.dropna(subset=['exit_signal'])
#   price_index = events.index.union(events['exit_signal']).drop_duplicates()
#   prices = close.reindex(price_index,method='bfill')

#   #2) create labels object
#   labels = pd.DataFrame(index=events.index)
#   labels['returns'] = prices.loc[events['exit_signal']].values/prices.loc[events.index]-1
#   labels['label'] = np.sign(labels['returns'])

#   return labels


# def compute_labels_after_algorithm(events, close):
#   """
#   Compute labels with metalabels (e.g. sides determined by a previous algorithm)
#   """
#   #1) prices aligned with events
#   events = events.copy()
#   events = events.dropna(subset=['vertical_barriers'])
#   prices = events.index.union(events['vertical_barriers']).drop_duplicates()
#   prices = close.reindex(prices, method='bfill')

#   #2) create return object
#   labels = pd.DataFrame(index=events.index)
#   entry_signals = events.index
#   exit_signals = events['exit_signal']

#   labels['returns'] = prices.loc[exit_signals].values / prices.loc[entry_signals] - 1
#   labels.loc[labels['returns']<=0,'label'] = 0

#   return labels