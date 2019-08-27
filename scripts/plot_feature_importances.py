#!/usr/bin/env python3
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "scripts"

import os
import ccxt
import json
import numpy as np
import warnings
import argparse
import scipy.stats as stats
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from utils.helpers import get_ohlcv_file, get_timeframe
from utils.scrape import scrape_ohlcv
from utils.bars import BAR_TYPES
from utils.csv import create_csv_files, open_convert_csv_files
from utils.cmd import default_parser
from utils.transforms import boxcox
from utils import from_exchange_to_standard_notation, from_standard_to_exchange_notation
import statsmodels.api as sm
from models import all_models
from scipy.cluster import hierarchy
from scipy.spatial import distance

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.graphics.gofplots import qqplot

from features.default import add_previous_returns
from features.default import add_returns
from features.default import add_lagged_returns
from features.targets import one_step_forward_returns

warnings.filterwarnings("ignore")


configuration_file = "./scripts/default_settings.json"
with open(configuration_file) as f:
  default_settings = json.load(f)

parser = default_parser()

parser.add_argument('-bt', '--bar_type',
                    type=str,
                    choices=list(BAR_TYPES.keys()))

parser.add_argument('-m', '--model',
                     type=str,
                     required=True,
                     choices=list(all_models.keys()))

parser.add_argument('-ic', '--include_correlations',
                    type=str,
                    nargs='+',
                    required=False,
                    help='Correlations you want to include')


args = parser.parse_args()

exchange_name = args.exchange or default_settings['default_exchange']
start = args.from_date or default_settings['default_start_date']
end = args.to_date or default_settings['default_end_date']
timeframe = args.timeframe or default_settings['default_timeframe']
symbol = args.symbols[0]
included_correlation_symbols = args.include_correlations
bar_type = BAR_TYPES[args.bar_type]
create_model = all_models[args.model]

bars = open_convert_csv_files(exchange_name, symbol, timeframe, start, end, bar_type=bar_type)

# Target Data
target_data = bars

# Feature Data
feature_data = {}
feature_data[symbol] = bars

if included_correlation_symbols:
  for s in included_correlation_symbols:
    feature_data[s] = open_convert_csv_files(exchange_name, s, timeframe, start, end, bar_type=bar_type)

X = pd.DataFrame()
y = one_step_forward_returns(target_data)

X, y = add_previous_returns(X, y, feature_data)
X, y = add_returns(X, y, feature_data)
X, y = add_lagged_returns(X, y, feature_data)

corr = X.corrwith(y)

# Plot correlations
plt.figure(figsize=(20,20))
corr.sort_values().plot.barh(color='blue', title='Strength of correlation')

# Plot orthogonality
plt.figure(figsize=(20,20))
corr_matrix = X.corr()
corr_array = np.asarray(corr_matrix)

linkage = hierarchy.linkage(distance.pdist(corr_array), \
                            method='average')

g = sns.clustermap(corr_matrix,row_linkage=linkage,col_linkage=linkage,\
                   row_cluster=True,col_cluster=True,figsize=(10,10),cmap='Greens')
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
label_order = corr_matrix.iloc[:,g.dendrogram_row.reordered_ind].columns

# Plot orthogonality for features above 0.1
correlated_features = corr[abs(corr)>0.05].index.tolist()

corr_matrix = X[correlated_features].corr()
corr_array = np.asarray(corr_matrix)

linkage = hierarchy.linkage(distance.pdist(corr_array), \
                            method='average')

g = sns.clustermap(corr_matrix,row_linkage=linkage,col_linkage=linkage,\
                   row_cluster=True,col_cluster=True,figsize=(6,6),cmap='Greens')
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

label_order = corr_matrix.iloc[:,g.dendrogram_row.reordered_ind].columns
print("Correlation Strength:")
print(corr[corr>0.05].sort_values(ascending=False))
plt.show()


# tmp = df[selected_features].join(outcome_scaled).reset_index().set_index('date')
# tmp.dropna().resample('Q').apply(lambda x: x.corr()).iloc[:,-1].unstack()\
# .iloc[:,:-1].plot(title='Correlation of Features to Outcome\n (by quarter)')
# # shows time stability