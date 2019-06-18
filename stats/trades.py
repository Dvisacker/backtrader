from collections import deque, OrderedDict

PNL_STATS = OrderedDict(
    [('Total profit', lambda x: x.sum()),
     ('Gross profit', lambda x: x[x > 0].sum()),
     ('Gross loss', lambda x: x[x < 0].sum()),
     ('Profit factor', lambda x: x[x > 0].sum() / x[x < 0].abs().sum()
      if x[x < 0].abs().sum() != 0 else np.nan),
     ('Avg. trade net profit', 'mean'),
     ('Avg. winning trade', lambda x: x[x > 0].mean()),
     ('Avg. losing trade', lambda x: x[x < 0].mean()),
     ('Ratio Avg. Win:Avg. Loss', lambda x: x[x > 0].mean() /
      x[x < 0].abs().mean() if x[x < 0].abs().mean() != 0 else np.nan),
     ('Largest winning trade', 'max'),
     ('Largest losing trade', 'min'),
     ])

SUMMARY_STATS = OrderedDict(
    [('Total number of round trips', 'count'),
     ('Percent profitable', lambda x: len(x[x > 0]) / float(len(x))),
     ('Winning round trips', lambda x: len(x[x > 0])),
     ('Losing round trips', lambda x: len(x[x < 0])),
     ('Even round trips', lambda x: len(x[x == 0])),
     ])

RETURN_STATS = OrderedDict(
    [('Avg returns all round trips', lambda x: x.mean()),
     ('Avg returns winning', lambda x: x[x > 0].mean()),
     ('Avg returns losing', lambda x: x[x < 0].mean()),
     ('Median returns all round trips', lambda x: x.median()),
     ('Median returns winning', lambda x: x[x > 0].median()),
     ('Median returns losing', lambda x: x[x < 0].median()),
     ('Largest winning trade', 'max'),
     ('Largest losing trade', 'min'),
     ])

DURATION_STATS = OrderedDict(
    [('Avg duration', lambda x: x.mean()),
     ('Median duration', lambda x: x.median()),
     ('Longest duration', lambda x: x.max()),
     ('Shortest duration', lambda x: x.min())
    ])

def agg_all_long_short(trades, col, stats_dict):
    stats_all = (trades
                 .assign(ones=1)
                 .groupby('ones')[col]
                 .agg(stats_dict)
                 .T
                 .rename_axis({1.0: 'All trades'},
                              axis='columns'))
    stats_long_short = (trades
                        .groupby('long')[col]
                        .agg(stats_dict)
                        .T
                        .rename_axis({False: 'Short trades',
                                      True: 'Long trades'},
                                     axis='columns'))

    return stats_all.join(stats_long_short)


def generate_trade_stats(trades):
    stats = {}
    stats['pnl'] = agg_all_long_short(trades, 'pnl', PNL_STATS)
    stats['summary'] = agg_all_long_short(trades, 'pnl', SUMMARY_STATS)
    stats['duration'] = agg_all_long_short(trades, 'duration', DURATION_STATS)
    stats['returns'] = agg_all_long_short(trades, 'rt_returns', RETURN_STATS)
    # stats['symbols'] = trades.groupby('symbol')['rt_returns'].agg(RETURN_STATS).T

    return stats
