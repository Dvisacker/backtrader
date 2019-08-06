from datetime import datetime

import numpy as np
import pandas as pd
import pdb

BAR_TYPES = {
  "time": "time_bars",
  "tick": "tick_bars",
  "volume": "contract_volume_bars",
  "base_volume": "base_volume_bars",
  "quote_volume": "quote_volume_bars",
  "imbalance": "flow_imbalance_bars"
}

TIME_FREQUENCIES = {
  "1m": "1T",
  "5m": "5T",
  "15m": "15T",
  "1h": "1H"
}

def get_candle_nb(days, timeframe='1m'):
  if (timeframe=='1m'):
    return 24 * 60 * days
  elif (timeframe=='5m'):
    return 25 * 60 * days / 5
  elif (timeframe=='15m'):
    return 25 * 60 * days / 15
  elif (timeframe=='1h'):
    return 25 * 60 * days / 15
  else:
    raise ValueError('Timeframe value not found')


class BarSeriesUtils(object):

  def __init__(self, days, timeframe):
    self.days = days
    self.timeframe = timeframe

  def get_recommended_tick_frequency(self, df):
    time_candle_nb = get_candle_nb(self.days, self.timeframe)
    total_trades = df.shape[0]
    return round(total_trades / time_candle_nb)

  def get_recommended_volume_frequency(self, df):
    time_candle_nb = get_candle_nb(self.days, self.timeframe)
    total_contract_volume = df['size'].sum()
    return total_contract_volume / time_candle_nb

  def get_recommended_base_currency_volume_frequency(self, df):
    time_candle_nb = get_candle_nb(self.days, self.timeframe)
    total_base_currency_volume = df['homeNotional'].sum()
    return total_base_currency_volume / time_candle_nb

  def get_recommended_quote_currency_volume_frequency(self, df):
    time_candle_nb = get_candle_nb(self.days, self.timeframe)
    total_quote_currency_volume = df['foreignNotional'].sum()
    return total_quote_currency_volume / time_candle_nb


class BarSeries(object):
    '''
        Base class for resampling ticks dataframe into OHLC(V)
        using different schemes. This particular class implements
        standard time bars scheme.
        See: https://www.wiley.com/en-it/Advances+in+Financial+Machine+Learning-p-9781119482086
    '''

    def __init__(self, df, datetimecolumn = 'time'):
        self.df = df
        self.datetimecolumn = datetimecolumn

    def process_column(self, column_name, frequency):
        return self.df[column_name].resample(frequency, label='right').ohlc()

    def process_volume(self, column_name, frequency):
        return self.df[column_name].resample(frequency, label='right').sum()

    def process_ticks(self, price_column = 'price', volume_column = 'size', frequency = '5T'):
        price_df = self.process_column(price_column, frequency)
        volume_df = self.process_volume(volume_column, frequency)
        price_df['volume'] = volume_df
        return price_df




class FlowImbalanceBarSeries(BarSeries):
    def __init__(self, quotes, ticks, datetimecolumn='time'):
        self.quotes = quotes
        self.ticks = ticks
        self.datetimecolumn = datetimecolumn

    def process_column(self, column_name, frequency):
        return self.ticks[column_name].resample(frequency, label='right').ohlc()

    def process_volume(self, column_name, frequency):
        return self.ticks[column_name].resample(frequency, label='right').sum()

    def process_ofi(self, frequency):
        quotes_df = self.quotes.copy().reset_index()
        quotes_df['midprice'] = ((quotes_df['bidPrice'] + quotes_df['askPrice']) / 2)
        quotes_df['spread'] = (quotes_df['bidPrice'] - quotes_df['askPrice'])
        quotes_df['midprice_returns'] = quotes_df['midprice'].diff()
        quotes_df['prevBidPrice'] = quotes_df['bidPrice'].shift()
        quotes_df['prevBidSize'] = quotes_df['bidSize'].shift()
        quotes_df['prevAskPrice'] = quotes_df['askPrice'].shift()
        quotes_df['prevAskSize'] = quotes_df['askSize'].shift()

        quotes_df.dropna(inplace=True)
        bid_geq = quotes_df['bidPrice'] >= quotes_df['prevBidPrice']
        bid_leq = quotes_df['bidPrice'] <= quotes_df['prevBidPrice']
        ask_geq = quotes_df['askPrice'] >= quotes_df['prevAskPrice']
        ask_leq = quotes_df['askPrice'] <= quotes_df['prevAskPrice']

        quotes_df['ofi'] = pd.Series(np.zeros(len(quotes_df)))
        quotes_df['ofi'].loc[bid_geq] += quotes_df['bidSize'].loc[bid_geq]
        quotes_df['ofi'].loc[bid_leq] -= quotes_df['prevBidSize'].loc[bid_leq]
        quotes_df['ofi'].loc[ask_geq] += quotes_df['prevAskSize'][ask_geq]
        quotes_df['ofi'].loc[ask_leq] -= quotes_df['askSize'][ask_leq]

        quotes_df = quotes_df.set_index('time')
        quotes_df = quotes_df[['midprice_returns','ofi']].resample(frequency).sum().dropna()
        return quotes_df

    def process_vfi(self, frequency):
        quotes_df = self.quotes.copy().reset_index()
        quotes_df['prevBidPrice'] = quotes_df['bidPrice'].shift()
        quotes_df['prevBidSize'] = quotes_df['bidSize'].shift()
        quotes_df['prevAskPrice'] = quotes_df['askPrice'].shift()
        quotes_df['prevAskSize'] = quotes_df['askSize'].shift()

        quotes_df.dropna(inplace=True)
        bid_geq = quotes_df['bidPrice'] > quotes_df['prevBidPrice']
        bid_eq = quotes_df['bidPrice'] == quotes_df['prevBidPrice']
        ask_leq = quotes_df['askPrice'] < quotes_df['prevAskPrice']
        ask_eq = quotes_df['askPrice'] == quotes_df['prevAskPrice']

        quotes_df['vfi'] = pd.Series(np.zeros(len(quotes_df)))
        quotes_df['vfi'].loc[bid_eq] += quotes_df['bidSize'].loc[bid_eq] - quotes_df['prevBidSize'].loc[bid_eq]
        quotes_df['vfi'].loc[bid_geq] += quotes_df['bidSize'].loc[bid_geq]
        quotes_df['vfi'].loc[ask_leq] -= quotes_df['askSize'].loc[ask_leq]
        quotes_df['vfi'].loc[ask_eq] += quotes_df['prevAskSize'].loc[ask_eq] - quotes_df['askSize'].loc[ask_eq]

        quotes_df = quotes_df.set_index('time')
        vfi = quotes_df['vfi'].resample(frequency).sum().dropna()
        return vfi


    def process_tfi(self, frequency):
        trades_df = self.ticks.copy()
        trades_df['signed_size'] = np.where(trades_df['side'] == 'Buy', trades_df['size'], -trades_df['size'])
        tfi = trades_df['signed_size'].resample(frequency).sum().dropna()
        return tfi

    def process_ticks(self, price_column='price', volume_column='size', frequency='5T'):
        price_df = self.process_column(price_column, frequency)
        volume_df = self.process_volume(volume_column, frequency)
        quotes_df = self.process_ofi(frequency)
        vfi = self.process_vfi(frequency)
        tfi = self.process_tfi(frequency)
        price_df['volume'] = volume_df
        price_df['ofi'] = quotes_df['ofi']
        price_df['vfi'] = vfi
        price_df['tfi'] = tfi
        price_df['midprice_returns'] = quotes_df['midprice_returns']

        return price_df


class TickBarSeries(BarSeries):
    '''
        Class for generating tick bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'time', volume_column = 'size'):
        self.volume_column = volume_column
        super(TickBarSeries, self).__init__(df, datetimecolumn)

    def process_column(self, column_name, frequency):
        res = []
        for i in range(frequency, len(self.df), frequency):
            sample = self.df.iloc[i-frequency:i]
            v = sample[self.volume_column].values.sum()
            o = sample[column_name].values.tolist()[0]
            h = sample[column_name].values.max()
            l = sample[column_name].values.min()
            c = sample[column_name].values.tolist()[-1]
            d = sample.index.values[-1]

            res.append({
                self.datetimecolumn: d,
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': v
            })

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res


    def process_ticks(self, price_column = 'price', volume_column = 'size', frequency = '5T'):
        price_df = self.process_column(price_column, frequency)
        return price_df


class VolumeBarSeries(BarSeries):
    '''
        Class for generating volume bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'time', volume_column = 'size'):
        self.volume_column = volume_column
        super(VolumeBarSeries, self).__init__(df, datetimecolumn)

    def process_column(self, column_name, frequency):
        res = []
        buf = []
        start_index = 0.
        volume_buf = 0.

        # pdb.set_trace()
        for i in range(len(self.df[column_name])):

            pi = self.df[column_name].iloc[i]
            vi = self.df[self.volume_column].iloc[i]
            di = self.df.index.values[i]

            buf.append(pi)
            volume_buf += vi

            if volume_buf >= frequency:

                o = buf[0]
                h = np.max(buf)
                l = np.min(buf)
                c = buf[-1]

                res.append({
                    self.datetimecolumn: di,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': volume_buf
                })

                buf, volume_buf = [], 0.

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res

    def process_ticks(self, price_column = 'price', volume_column = 'size', frequency = 10000):
        price_df = self.process_column(price_column, frequency)
        return price_df


class QuoteCurrencyVolumeBarSeries(BarSeries):
    '''
        Class for generating "dollar" bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'time', volume_column = 'size'):
        self.volume_column = volume_column
        super(QuoteCurrencyVolumeBarSeries, self).__init__(df, datetimecolumn)

    def process_column(self, column_name, frequency):
        res = []
        buf, vbuf = [], []
        start_index = 0.
        quote_currency_volume_buf = 0.
        for i in range(len(self.df[column_name])):

            pi = self.df[column_name].iloc[i]
            vi = self.df[self.volume_column].iloc[i]
            di = self.df.index.values[i]
            dvi = self.df['foreignNotional'].iloc[i]
            buf.append(pi)
            vbuf.append(vi)
            quote_currency_volume_buf += dvi

            if quote_currency_volume_buf >= frequency:

                o = buf[0]
                h = np.max(buf)
                l = np.min(buf)
                c = buf[-1]
                v = np.sum(vbuf)

                res.append({
                    self.datetimecolumn: di,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': v,
                    'quote_currency_volume': quote_currency_volume_buf
                })

                buf, vbuf, quote_currency_volume_buf = [], [], 0.

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res

    def process_ticks(self, price_column = 'price', volume_column = 'size', frequency = 10000):
        price_df = self.process_column(price_column, frequency)
        return price_df



class BaseCurrencyVolumeBarSeries(BarSeries):
    '''
        Class for generating "dollar" bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'time', volume_column = 'size'):
        self.volume_column = volume_column
        super(BaseCurrencyVolumeBarSeries, self).__init__(df, datetimecolumn)

    def process_column(self, column_name, frequency):
        res = []
        buf, vbuf = [], []
        start_index = 0.
        base_currency_buf = 0.
        for i in range(len(self.df[column_name])):

            pi = self.df[column_name].iloc[i]
            vi = self.df[self.volume_column].iloc[i]
            di = self.df.index.values[i]
            dvi = self.df['homeNotional'].iloc[i]
            buf.append(pi)
            vbuf.append(vi)
            base_currency_buf += dvi

            if base_currency_buf >= frequency:

                o = buf[0]
                h = np.max(buf)
                l = np.min(buf)
                c = buf[-1]
                v = np.sum(vbuf)

                res.append({
                    self.datetimecolumn: di,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': v,
                    'base_currency_volume': base_currency_buf
                })

                buf, vbuf, base_currency_buf = [], [], 0.

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res

    def process_ticks(self, price_column = 'price', volume_column = 'size', frequency = 10000):
        price_df = self.process_column(price_column, frequency)
        return price_df



class ImbalanceTickBarSeries(BarSeries):
    '''
        Class for generating imbalance tick bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'time', volume_column = 'size'):
        self.volume_column = volume_column
        super(ImbalanceTickBarSeries, self).__init__(df, datetimecolumn)

    def get_bt(self, data):
        s = np.sign(np.diff(data))
        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i-1]
        return s

    def get_theta_t(self, bt):
        return np.sum(bt)

    def ewma(self, data, window):

        alpha = 2 /(window + 1.0)
        alpha_rev = 1-alpha

        scale = 1/alpha_rev
        n = data.shape[0]

        r = np.arange(n)
        scale_arr = scale**r
        offset = data[0]*alpha_rev**(r+1)
        pw0 = alpha*alpha_rev**(n-1)

        mult = data*pw0*scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums*scale_arr[::-1]
        return out

    def process_column(self, column_name, initital_T = 100, min_bar = 10, max_bar = 1000):
        init_bar = self.df[:initital_T][column_name].values.tolist()

        ts = [initital_T]
        bts = [bti for bti in self.get_bt(init_bar)]
        res = []

        buf_bar, vbuf, T = [], [], 0.
        for i in range(initital_T, len(self.df)):


            di = self.df.index.values[i]

            buf_bar.append(self.df[column_name].iloc[i])
            bt = self.get_bt(buf_bar)
            theta_t = self.get_theta_t(bt)

            try:
                e_t = self.ewma(np.array(ts), initital_T / 10)[-1]
                e_bt = self.ewma(np.array(bts), initital_T)[-1]
            except:
                e_t = np.mean(ts)
                e_bt = np.mean(bts)
            finally:
                if np.isnan(e_bt):
                    e_bt = np.mean(bts[int(len(bts) * 0.9):])
                if np.isnan(e_t):
                    e_t = np.mean(ts[int(len(ts) * 0.9):])


            condition = np.abs(theta_t) >= e_t * np.abs(e_bt)


            if (condition or len(buf_bar) > max_bar) and len(buf_bar) >= min_bar:

                o = buf_bar[0]
                h = np.max(buf_bar)
                l = np.min(buf_bar)
                c = buf_bar[-1]
                v = np.sum(vbuf)

                res.append({
                    self.datetimecolumn: di,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': v
                })

                ts.append(T)
                for b in bt:
                    bts.append(b)

                buf_bar = []
                vbuf = []
                T = 0.
            else:
                vbuf.append(self.df[self.volume_column].iloc[i])
                T += 1

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res

    def process_ticks(self, price_column = 'price', volume_column = 'size', init = 100, min_bar = 10, max_bar = 1000):
        price_df = self.process_column(price_column, init, min_bar, max_bar)
        return price_df