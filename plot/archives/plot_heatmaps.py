import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_data_matrix(df, x_axis, y_axis, value_axis):
  nb_rows = len(set(df[x_axis]))
  nb_columns = len(set(df[y_axis]))
  data = np.zeros((nb_rows, nb_columns))

  row_labels = df[x_axis]
  column_labels = df[y_axis]

  for i in range(0, nb_rows):
    for j in range(0, nb_columns):
      data[i][j] = float(df[value_axis][nb_columns * i + j])

  return data


def create_all_data_matrixes(df, x_axis, y_axis, value_axes, variable_param=None):
  nb_rows = len(set(df[x_axis]))
  nb_columns = len(set(df[y_axis]))
  row_labels = df[x_axis]
  column_labels = df[y_axis]

  if (variable_param == None):
    result = {}
    for k in value_axes:
      data = np.zeros((nb_rows, nb_columns))
      for i in range(0, nb_rows):
        for j in range(0, nb_columns):
          data[i][j] = float(df[k][nb_columns * i + j])

      result[k] = data
    return [result]

  nb_variable_param = len(set(df[variable_param]))
  matrixes = {}
  for l in range(0, nb_variable_param):
    result = {}
    for k in value_axes:
      data = np.zeros((nb_rows, nb_columns))
      for i in range(0, nb_rows):
        for j in range(0, nb_columns):
          data[i][j] = float(df[k][nb_columns * nb_rows * l + nb_columns * i + j])

      result[k] = data
    matrixes[df[variable_param][nb_columns * nb_rows * l]] = result

  return matrixes


def generate_heatmaps():
  df = pd.read_csv('opt.csv')
  indicators = ['Total Returns', 'Sharpe Ratio', 'Max Drawdown', 'Drawdown Duration']
  indicators_colormaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Oranges, plt.cm.Greens]
  x_axis = 'zscore_entry'
  y_axis = 'zscore_exit'
  variable_param = 'ols_window'

  row_labels = df[x_axis]
  column_labels = df[y_axis]
  matrixes = create_all_data_matrixes(df, x_axis, y_axis, indicators, variable_param=variable_param)

  for i in range(0, len(indicators)):
    for k in list(matrixes.keys()):
      indicator = indicators[i]
      data = matrixes[k][indicator]
      fig, ax = plt.subplots()
      heatmap = ax.pcolor(data, cmap=indicators_colormaps[i])

      for y in range(data.shape[0]):
        for x in range(data.shape[1]):
          plt.text(x + 0.5, y + 0.5, "%.2f%%" % data[y, x],
            horizontalalignment="center",
            verticalalignment="center",
          )

      plt.colorbar(heatmap)
      ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
      ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
      ax.set_xticklabels(row_labels, minor=False)
      ax.set_yticklabels(column_labels, minor=False)
      plt.suptitle('{} - {}: {}'.format(indicator, variable_param, k), fontsize=18)
      plt.xlabel(x_axis, fontsize=14)
      plt.ylabel(y_axis, fontsize=14)

  plt.show()






# def output_heatmaps(self):
#   """
#   Creates a list of heatmaps that show the impact of different parameters
#   on backtest results
#   """
#   df = pd.read_csv('opt.csv')

#   data = np.zeros

#   csv_file = open('results/opt.csv', 'r').readlines()
#   csv_ref = [ c.strip().split(",") for c in csv_file if c[:3] == "100" ]

#   data = _create_data_matrix(csv_ref, 3)
#   fig, ax = plt.subplots()
#   heatmap =

#   g = sns.Face
