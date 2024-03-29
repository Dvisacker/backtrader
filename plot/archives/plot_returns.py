#!/usr/bin/python
# -*- coding: utf-8 -*-
# plot_sharpe.py

import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
  # Open the CSV file and obtain only the lines
  # with a lookback value of 100
  csv_file = open("results/last/opt.csv", "r").readlines()
  csv_ref = [ c.strip().split(",") for c in csv_file if c[:3] == "100" ]
  data = create_data_matrix(csv_ref, 3)
  fig, ax = plt.subplots()
  heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
  row_labels = [0.5, 1.0, 1.5]
  column_labels = [2.0, 3.0, 4.0]

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
  plt.suptitle("Return Ratio Heatmap", fontsize=18)
  plt.xlabel("Z-Score Exit Threshold", fontsize=14)
  plt.ylabel("Z-Score Entry Threshold", fontsize=14)
  plt.show()