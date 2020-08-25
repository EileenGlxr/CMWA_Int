# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 09:13:57 2020

@author: liuxr

策略说明
    1.不可以平今仓
    2.换仓 - 收盘换仓，可在交割日前0,1,2,3,日收盘换仓。

"""


import pandas as pd
import numpy as np
import datetime as dt
import sys
import os


import load_data


# 全局变量记录各种参数
class Params():
    def __init__(self, code, start_date, switch_ahead_days):
        self.big_bar_percent = 2.5
        self.start_date = start_date
        self.count_bar = 6
        self.last_day_diff = 1
        self.ma_length = 5
        self.delay_day = 2
        self.length1 = 9
        self.length2 = 6
        self.length = 4

        self.switch_ahead_days = switch_ahead_days

        self.initial_amount = None  # float,初始资金
        self.amount = None  # float,记录累计收益
        if code == 'IH' or 'IF':
            self.mul = 300  # 期货倍数，IF为300
        elif code == 'IC':
            self.mul = 200

        self.code = code

        self.settle_dates = None  # dict，合约及对应交割日
        self.trading_days = None  # list

        self.dict_30m = dict()  # StockID - dataframe

        self.open_day = None
        self.position = None  # str, 'long' or 'short'
        self.contract = None  # str, 持有的合约
        self.contract_considered = None
        self.contract_num = 1  # int,持有的手数。本策略中始终为1
        self.trade_win = None
        self.daily_win = None
        self.netvalue = None  # float，每日收盘计算净值
        self.price = None
        self.fee = 0.000023  # 单边交易手续费，万分之0.23

        self.entrybar_iid = None

        self.trail_start = 1
        self.trail_stop = 80
        self.ATRK = 3

    def split_dataframe(self, df_30m):
        contracts = df_30m.StockID.unique().tolist()
        for con in contracts:
            self.dict_30m[con] = df_30m.loc[df_30m['StockID'] == con]

    def jdiff_atr_calculation(self):

        for con, df in self.dict_30m.items():
            df.insert(len(df.columns), 'JDiff', None)
            df.insert(len(df.columns), 'ATR', None)
            df.insert(len(df.columns), 'upper_band', None)
            df.insert(len(df.columns), 'lower_band', None)

            low_list = df['low'].rolling(self.length1, min_periods=self.length1).min()
            low_list.fillna(value=df['low'].expanding().min(), inplace=True)
            high_list = df['high'].rolling(self.length1, min_periods=self.length1).max()
            high_list.fillna(value=df['high'].expanding().max(), inplace=True)

            rsv1 = 100 * (df['high'] - low_list) / (high_list - low_list)
            k1 = rsv1.ewm(com=self.length2).mean()
            d1 = k1.ewm(com=self.length2).mean()
            j1 = 3 * k1 - 2 * d1

            rsv2 = 100 * (high_list - df['low']) / (high_list - low_list)
            k2 = rsv2.ewm(com=self.length2).mean()
            d2 = k2.ewm(com=self.length2).mean()
            j2 = 3 * k2 - 2 * d2

            df.loc[:, 'JDiff'] = j1 - j2

            TR = pd.concat(
                [df['high'] - df['low'], abs(df['close'].shift(1) - df['high']), abs(df['close'].shift(1) - df['low'])],
                axis=1)
            df['ATR'] = TR.max(axis=1).rolling(14).mean()

            df['upper_band'] = df['high'].rolling(self.length).max()
            df['lower_band'] = df['low'].rolling(self.length).min()

            df.reset_index(drop=True, inplace=True)

    def update_tradingdays(self, tradingdays):
        self.trading_days = tradingdays

    def update_settledates(self, df_olhc):
        self.settle_dates = dict()
        df_unique = df_olhc.drop_duplicates(['StockID'])
        for ind in df_unique.index:
            self.settle_dates[df_unique['StockID'].loc[ind]] = df_unique['SettleDate'].loc[ind]


# %%

# 返回日线状态判断
def trend_day_mark(df_daily_olhc):
    global P

    tradingdays = list(df_daily_olhc.Date.unique())
    tradingdays.sort()
    P.update_tradingdays(tradingdays)
    P.update_settledates(df_daily_olhc)

    df_trend = pd.DataFrame(index=tradingdays,
                            columns=['curr_con', 'curr_con_trend', 'curr_con_trand2', 'next_con', 'next_con_trend',
                                     'next_con_trend2'])

    '''
    First method to evaluate the trend
    '''
    for day in tradingdays:
        iid = tradingdays.index(day)

        daily_data = df_daily_olhc.loc[df_daily_olhc['Date'] == day]

        contracts = daily_data.StockID.unique()
        contracts.sort()

        df_trend.loc[day, 'curr_con'] = contracts[0]
        df_trend.loc[day, 'next_con'] = contracts[1]
        curr_con_data = daily_data.loc[daily_data['StockID'] == contracts[0]].iloc[0]
        next_con_data = daily_data.loc[daily_data['StockID'] == contracts[1]].iloc[0]

        curr_con_daily_change = (curr_con_data['close'] - curr_con_data['open']) / curr_con_data['open']
        if curr_con_daily_change >= P.big_bar_percent * 0.01:
            df_trend.loc[day, 'curr_con_trend'] = 1
        elif curr_con_daily_change <= (-1) * P.big_bar_percent * 0.01:
            df_trend.loc[day, 'curr_con_trend'] = -1

        next_con_daily_change = (next_con_data['close'] - next_con_data['open']) / next_con_data['open']
        if next_con_daily_change >= P.big_bar_percent * 0.01:
            df_trend.loc[day, 'next_con_trend'] = 1
        elif next_con_daily_change <= (-1) * P.big_bar_percent * 0.01:
            df_trend.loc[day, 'next_con_trend'] = -1

    '''
    Second method to evaluate the trend
    '''
    df_MA_record = pd.DataFrame(columns=['Date', 'StockID', 'MA' + str(P.ma_length), 'bigger_dummy', 'lesser_dummy'])

    for day in tradingdays[P.ma_length:]:
        iid = tradingdays.index(day)
        period_data = df_daily_olhc.loc[df_daily_olhc['Date'].isin(tradingdays[iid - P.ma_length + 1:iid + 1])]

        curr_MA = period_data.loc[period_data['StockID'] == df_trend.loc[day, 'curr_con']]['close'].mean()
        next_MA = period_data.loc[period_data['StockID'] == df_trend.loc[day, 'next_con']]['close'].mean()
        curr_close = \
            period_data.loc[(period_data['StockID'] == df_trend.loc[day, 'curr_con']) & (period_data['Date'] == day)][
                'close'].iloc[0]
        next_close = \
            period_data.loc[(period_data['StockID'] == df_trend.loc[day, 'next_con']) & (period_data['Date'] == day)][
                'close'].iloc[0]

        curr_row = pd.DataFrame(
            [[day, df_trend.loc[day, 'curr_con'], curr_MA, curr_close > curr_MA, curr_close < curr_MA]],
            columns=['Date', 'StockID', 'MA' + str(P.ma_length), 'bigger_dummy', 'lesser_dummy'])

        next_row = pd.DataFrame(
            [[day, df_trend.loc[day, 'next_con'], next_MA, next_close > next_MA, next_close < next_MA]],
            columns=['Date', 'StockID', 'MA' + str(P.ma_length), 'bigger_dummy', 'lesser_dummy'])

        df_MA_record = pd.concat([df_MA_record, curr_row, next_row], axis=0)

    df_MA_record.reset_index()

    for day in tradingdays[P.ma_length:]:
        iid = tradingdays.index(day)

        curr_con = df_trend.loc[day, 'curr_con']
        MA_row = df_MA_record.loc[(df_MA_record['Date'] == day) & (df_MA_record['StockID'] == curr_con)].iloc[0]
        MA = MA_row['MA' + str(P.ma_length)]

        df_curr_con = df_MA_record.loc[df_MA_record['StockID'] == curr_con]
        df_curr_con = df_curr_con.loc[df_curr_con['Date'] <= day]

        if len(df_curr_con) < P.count_bar:
            pass
        else:
            df_curr_con = df_curr_con.iloc[-P.count_bar:]
            curr_close = \
                df_daily_olhc.loc[(df_daily_olhc['Date'] == day) & (df_daily_olhc['StockID'] == curr_con)][
                    'close'].iloc[0]

            if df_curr_con['bigger_dummy'].sum() >= P.count_bar and curr_close >= MA * (1 + P.last_day_diff * 0.01):
                df_trend.loc[day, 'curr_con_trend2'] = 1

            if df_curr_con['lesser_dummy'].sum() >= P.count_bar and curr_close <= MA * (1 + P.last_day_diff * 0.01):
                df_trend.loc[day, 'curr_con_trend2'] = -1

        # next_con再来一遍
        next_con = df_trend.loc[day, 'next_con']
        MA_row = df_MA_record.loc[(df_MA_record['Date'] == day) & (df_MA_record['StockID'] == next_con)].iloc[0]
        MA = MA_row['MA' + str(P.ma_length)]

        df_next_con = df_MA_record.loc[df_MA_record['StockID'] == next_con]
        df_next_con = df_next_con.loc[df_next_con['Date'] <= day]

        if len(df_next_con) < P.count_bar:
            pass
        else:
            df_next_con = df_next_con.iloc[-P.count_bar:]
            next_close = \
                df_daily_olhc.loc[(df_daily_olhc['Date'] == day) & (df_daily_olhc['StockID'] == next_con)][
                    'close'].iloc[0]

            if df_next_con['bigger_dummy'].sum() >= P.count_bar and next_close >= MA * (1 + P.last_day_diff * 0.01):
                df_trend.loc[day, 'next_con_trend2'] = 1

            if df_next_con['lesser_dummy'].sum() >= P.count_bar and next_close <= MA * (1 + P.last_day_diff * 0.01):
                df_trend.loc[day, 'curr_con_trend2'] = -1

    return df_trend


# 下单函数
def make_order(day, time, contract, signal, price, entrybar_iid=None):
    global P

    global df_win
    global df_make_order_record

    if P.initial_amount == None:
        P.initial_amount = price * P.mul
        P.amount = P.initial_amount

    if signal == 'close_pos':  # 只有平仓的时候更新amount
        if day == P.open_day:
            fee_pct = P.fee * 15
        else:
            fee_pct = P.fee

        if P.position == 'long':  # 平多仓
            fee = fee_pct * P.mul * price + fee_pct * P.mul * P.price
            profit = (price - P.price) * P.mul - fee
            P.amount = P.amount + (price - P.price) * P.mul  # 平仓价格减去开仓价格
            P.contract = None
            P.position = None
            P.trade_win = True if price > P.price else False
            P.price = None
            P.entry_bar_iid = None

            P.open_day = None

        elif P.position == 'short':  # 平空仓
            fee = fee_pct * P.mul * (P.price + price)
            profit = (P.price - price) * P.mul - fee
            P.amount = P.amount + (P.price - price) * P.mul  # 开仓价格减去平仓价格
            P.contract = None
            P.position = None
            P.trade_win = True if P.price > price else False
            P.price = None
            P.entry_bar_iid = None
            P.open_day = None

        row = pd.DataFrame([[P.trade_win, fee]], index=[day], columns=['Win', 'transaction_fee'])
        df_win = pd.concat([df_win, row], axis=0)

    if signal == 'long':  # 开多仓
        P.contract = contract
        P.contract_num = 1
        P.position = 'long'
        P.price = price  # 开仓时的做多价
        profit = None
        P.entrybar_iid = entrybar_iid
        P.open_day = day

    if signal == 'short':  # 开空仓
        P.contract = contract
        P.contract_num = 1
        P.position = 'short'
        P.price = price  # 开仓时的做空价
        profit = None
        P.entrybar_iid = entrybar_iid
        P.open_day = day

    row = pd.DataFrame([[day, time, contract, signal, price, profit, P.amount]],
                       columns=['Date', 'Time', 'FutureID', 'Action', 'Price', 'Profit', 'Amount'])

    df_make_order_record = pd.concat([df_make_order_record, row], axis=0)


# 收盘前更新净值
def update_before_close(day, df_daily_olhc, df_trend):
    global P
    global df_result
    today_trade_data = df_daily_olhc.loc[df_daily_olhc['Date'] == day]

    curr_con = df_trend.loc[day, 'curr_con']

    if P.contract == None:
        stlmnt_date = P.settle_dates[curr_con]
        days_to_maturity = dt.datetime.strptime(str(stlmnt_date), '%Y%m%d') - dt.datetime.strptime(str(day), '%Y%m%d')

        if days_to_maturity.days <= P.switch_ahead_days:
            P.contract_considered = df_trend.loc[day, 'next_con']  # 更新考虑的合约

        P.netvalue = P.amount
        P.daily_win = None

    else:
        stlmnt_date = P.settle_dates[P.contract]
        days_to_maturity = dt.datetime.strptime(str(stlmnt_date), '%Y%m%d') - dt.datetime.strptime(str(day), '%Y%m%d')

        if days_to_maturity.days <= P.switch_ahead_days:  # 3天，周二收盘换

            close_price = today_trade_data['close'].loc[today_trade_data['StockID'] == P.contract].iloc[0]
            make_order(day, 'close_switch', P.contract, 'close_pos', close_price)  # 平

            P.contract_considered = df_trend.loc[day, 'next_con']  # 更新考虑的合约


        # =============================================================================
        #             open_price = today_trade_data['open'].loc[today_trade_data['StockID'] == P.contract_considered].iloc[0]
        #             df_30m_con = P.dict_30m[P.contract_considered]
        #             iid = df_30m_con.loc[df_30m_con['Date'] == day].index[0]
        #             make_order(day, 'open_switch', P.contract_considered, signal, open_price, iid)  # 开
        # =============================================================================
        else:
            trade_data = \
                df_daily_olhc.loc[(df_daily_olhc['Date'] == day) & (df_daily_olhc['StockID'] == P.contract)].iloc[0]

            if P.position == 'short':
                profit = P.contract_num * (P.price - trade_data['close'])  # 做空价格-收盘价
                P.netvalue = P.amount + profit

            elif P.position == 'long':
                profit = P.contract_num * (trade_data['close'] - P.price)  # 做空价格-收盘价
                P.netvalue = P.amount + profit

            if P.position == 'long' and trade_data['close'] > trade_data['open']:
                P.daily_win = True

            elif P.position == 'short' and trade_data['close'] < trade_data['open']:
                P.daily_win = True

            else:
                P.daily_win = False

    row = pd.DataFrame([[P.contract, P.contract_num, P.position, P.netvalue, P.daily_win]],
                       index=[day],
                       columns=['FutureID', 'ContractNum', 'Position', 'Value', 'Win'])

    df_result = pd.concat([df_result, row], axis=0)


# 确认当天市场状况
def trend_confirmation(day, df_trend):
    global P

    iid = df_trend.index.tolist().index(day)
    trend_row = df_trend.loc[day]
    curr_next = 'curr_' if P.contract_considered == trend_row['curr_con'] else 'next_'

    signal1 = df_trend.iloc[max(0, iid - P.delay_day):iid][curr_next + 'con_trend']
    signal2 = df_trend.iloc[iid - 1:iid][curr_next + 'con_trend2']

    if signal1.isna().sum() == len(signal1) and signal2.isna().sum():  # 都没有信号
        return 0
    elif signal1.isna().sum() < len(signal1):  # 方法一优先
        return signal1.loc[~signal1.isna()].iloc[-1]
    else:  # 只有方法二有信号，提取方法二的信号
        return signal2.iloc[0]


# 日内每个bar进行监控
def intraday_track(day, today_trend):
    global P

    bars = P.dict_30m[P.contract_considered]

    tradetime = bars.loc[bars['Date'] == day]['Time'].unique()
    tradetime.sort()

    # 当日进行交易
    for time in tradetime:
        this_row = bars.loc[(bars['Date'] == day) & (bars['Time'] == time)]
        iid = bars.index.tolist().index(this_row.index)

        if iid < P.length1 + P.length2:  # 不足以计算，不进行
            continue

        # 非强下跌，满足条件平空开多
        if today_trend != -1 and bars['JDiff'].iloc[iid - 1] >= 0 and bars['high'].iloc[iid] >= bars['upper_band'].iloc[
            iid - 1] and bars['close'].iloc[iid - 1] >= bars['high'].iloc[iid - 2]:
            if P.position == 'short':
                if day != P.open_day:  # 不平今仓
                    make_order(day, time, P.contract, 'close_pos',
                               max(bars['open'].iloc[iid], bars['upper_band'].iloc[iid - 1]))

                    make_order(day, time, P.contract_considered, 'long',
                               max(bars['open'].iloc[iid], bars['upper_band'].iloc[iid - 1]), iid)
                else:
                    pass

            elif P.position == None:
                make_order(day, time, P.contract_considered, 'long',
                           max(bars['open'].iloc[iid], bars['upper_band'].iloc[iid - 1]), iid)
            else:
                pass

        # 非强上涨，满足条件平多开空
        if today_trend != 1 and bars['JDiff'].iloc[iid - 1] >= 0 and bars['low'].iloc[iid] <= bars['lower_band'].iloc[
            iid - 1] and bars['close'].iloc[iid - 1] <= bars['low'].iloc[iid - 2]:
            if P.position == 'long':
                if day != P.open_day:  # 不平今仓
                    make_order(day, time, P.contract, 'close_pos',
                               min(bars['open'].iloc[iid], bars['lower_band'].iloc[iid - 1]))

                    make_order(day, time, P.contract_considered, 'short',
                               min(bars['open'].iloc[iid], bars['lower_band'].iloc[iid - 1]), iid)
                else:
                    pass
            elif P.position == None:
                make_order(day, time, P.contract_considered, 'short',
                           min(bars['open'].iloc[iid], bars['lower_band'].iloc[iid - 1]), iid)
            else:
                pass

        # 非空仓止损止盈
        if P.contract:
            bars_since_entry = iid - P.entrybar_iid

            temp_high = bars['high'].iloc[P.entrybar_iid:iid].max()
            temp_low = bars['low'].iloc[P.entrybar_iid:iid].min()

            if P.position == 'long' and bars_since_entry > 0:
                if temp_high > P.price * (1 + P.trail_start * 0.01):  # 止盈
                    stop_line = temp_high - (temp_high - P.price) * P.trail_stop * 0.01
                    if bars['low'].iloc[iid] < stop_line and day != P.open_day:
                        make_order(day, time, P.contract, 'close_pos', min(stop_line, bars['open'].iloc[iid]))

                elif temp_high <= P.price * (1 + P.trail_start * 0.01):  # 止损
                    stop_line = P.price - bars['ATR'].iloc[iid - 1] * P.ATRK
                    if bars['low'].iloc[iid] < stop_line and day != P.open_day:
                        make_order(day, time, P.contract, 'close_pos', min(stop_line, bars['open'].iloc[iid]))

            if P.position == 'short' and bars_since_entry > 0:
                if temp_low < P.price * (1 - P.trail_start * 0.01):  # 止盈
                    stop_line = temp_low + (P.price - temp_low) * P.trail_stop * 0.01
                    if bars['high'].iloc[iid] > stop_line and day != P.open_day:
                        make_order(day, time, P.contract, 'close_pos', max(stop_line, bars['open'].iloc[iid]))

                elif temp_low >= P.price * (1 - P.trail_start * 0.01):  # 止盈
                    stop_line = P.price + bars['ATR'].iloc[iid - 1] * P.ATRK
                    if bars['high'].iloc[iid] > stop_line and day != P.open_day:
                        make_order(day, time, P.contract, 'close_pos', max(stop_line, bars['open'].iloc[iid]))


def run_daily(day, df_daily_olhc, df_trend):
    global P

    if P.contract_considered == None:
        P.contract_considered = df_trend.loc[day, 'curr_con']

    today_trend = trend_confirmation(day, df_trend)  # 确定当天是否处于强上涨、强下跌或震荡
    intraday_track(day, today_trend)

    update_before_close(day, df_daily_olhc, df_trend)


# %%
# 计算最大回撤函数
def calcmaxdd(x):
    '''
    :param x: 净值序列
    :return: maxdd
    '''
    tempx = x.dropna()
    if tempx.shape[0] > 1:
        tempmin = tempx.min()
        tempminindex = np.where(tempx == tempmin)[0][-1]
        tempmax = tempx.iloc[:tempminindex + 1].max()  # 最低点之前的max
        maxdd = 1 - tempmin / tempmax
        while tempx.iloc[tempminindex + 1:].max() > tempmax:
            tempmin = tempx.iloc[tempminindex + 1:].min()
            tempminindex1 = np.where(tempx == tempmin)[0][-1]
            tempmax = tempx.iloc[tempminindex + 1:tempminindex1 + 1].max()
            tempmaxdd = 1 - tempmin / tempmax
            tempminindex = tempminindex1
            if tempmaxdd > maxdd:
                maxdd = tempmaxdd

        return maxdd

    else:  # 只有一个值
        return np.nan


def summary(df_result, df_win):
    df_result_sum=df_result.dropna()
    df_result_sum['daily_return'] = df_result_sum['Value'].pct_change(1)  # series
    df_summary = pd.DataFrame(
        columns=['annual_return', 'annual_return_std', 'return_std_ratio', 'daily_win_ratio', 'trade_win_ratio',
                 'maxdd',
                 'total_fee_ratio', 'total_trading_days', 'total_switching_times'])

    #2010开始，分年统计
    if P.code == 'IF':
        three_year_end = 20130418
        five_year_end = 20150422
        end = int(P.trading_days[-1])
    else:
        three_year_end = 20180420
        five_year_end = 20200416
        end = int(P.trading_days[-1])
        
    for end_day in [three_year_end, five_year_end, end]:
        df_result_used = df_result_sum.loc[:end_day + 1]
        df_win_used = df_win.loc[:end_day + 1]
        # 年化收益率
        print('<-----end_day-------->',end_day)
        df_result_used['net_value_1'] = df_result_used['Value'] / df_result_used['Value'].iloc[0]
        # annual_return = pow(df_result_used['Value'].iloc[-1] / df_result_used['Value'].iloc[0], 1 / 3) - 1  # float
        tempinterval = dt.datetime.strptime(str(df_result_used.index[-1]), "%Y%m%d") - dt.datetime.strptime(
            str(df_result_used.index[0]),
            "%Y%m%d")
        intervaldays = tempinterval.days
        intervalyears = intervaldays / 365
        annual_return = np.exp(np.log(df_result_used['net_value_1'].iloc[-1]) / intervalyears) - 1  # 年化收益率

        annual_return_std = np.std(df_result_used['daily_return']) * np.sqrt(252)  # 波动率
        return_std_ratio = annual_return / annual_return_std  # float

        # 日度胜率
        daily_win_ratio = len(df_result_used.loc[df_result_used['Win'] == True]) / len(df_result_used)  # float

        # 交易胜率
        trades_win_ratio = sum(df_win_used['Win'] == True) / len(df_win_used)

        # 手续费占比
        total_fee = df_win_used['transaction_fee'].sum()
        total_fee_ratio = total_fee / (
                df_result_used['Value'].iloc[-1] - df_result_used['Value'].iloc[0] + total_fee)

        # 最大回撤
        maxdd = calcmaxdd(df_result_used['Value'])

        # 交易天数
        total_trading_days = len(df_result_used.dropna())

        # 换仓次数
        total_switching_times = len(df_win_used)

        row = pd.DataFrame(
            [[annual_return, annual_return_std, return_std_ratio, daily_win_ratio, trades_win_ratio, maxdd,
              total_fee_ratio,
              total_trading_days, total_switching_times]], index=[end_day],
            columns=['annual_return', 'annual_return_std', 'return_std_ratio', 'daily_win_ratio', 'trade_win_ratio',
                     'maxdd',
                     'total_fee_ratio', 'total_trading_days', 'total_switching_times'])

        df_summary = pd.concat([df_summary, row], axis=0)



    return df_summary


# %%

if __name__ == '__main__':
    currentpath=os.path.dirname(__file__)
    sys.path.append(currentpath)

    FuturesCodeList =['IF','IH','IC']
    SwitchAheadList = [0, 1, 2, 3]

    startdate = 20100101
    enddate = 20200717

    for code in FuturesCodeList:
        for switch_ahead_day in SwitchAheadList:
            dir_path = 'data/'

            df_30m_path = code + '_30m_' + str(20100101) + '_' + str(enddate) + '.csv'
            df_daily_olhc_path = code + 'DailyTradeData_' + str(20100101) + '_' + str(enddate) + '.csv'

            if os.path.exists(dir_path + df_daily_olhc_path):
                df_daily_olhc = pd.read_csv(dir_path + df_daily_olhc_path)
                df_30m = pd.read_csv(dir_path + df_30m_path)
                print('data_read_from_files')

            else:
                df_daily_olhc = load_data.getFuturesdata_daily(code, startdate, enddate)
                df_30m = load_data.getFuturesdata_30min(code, startdate, enddate)
                print('data loaded from TinySoft')
                df_daily_olhc.to_csv(dir_path + df_daily_olhc_path)
                df_30m.to_csv(dir_path + df_30m_path)
            
            res_dir = 'result/' + str(startdate) + '_' + str(
                enddate) + '/' + code + '/switch_' + str(switch_ahead_day) + '_ahead/'
            if os.path.exists(res_dir):
                print('result exists!')
                continue

            # 20170701至今
            df_daily_olhc = df_daily_olhc.loc[df_daily_olhc['Date'] >= startdate]
            df_30m = df_30m.loc[df_30m['Date'] >= startdate]

            P = Params(code, startdate, switch_ahead_day)
            P.split_dataframe(df_30m)
            P.jdiff_atr_calculation()

            df_result = pd.DataFrame(columns=['FutureID', 'ContractNum', 'Position', 'Value', 'Win'])
            df_win = pd.DataFrame(columns=['Win', 'transaction_fee'])
            df_make_order_record = pd.DataFrame(
                columns=['Date', 'Time', 'FutureID', 'Action', 'Price', 'Profit', 'Amount'])

            df_trend = trend_day_mark(df_daily_olhc)
            for day in P.trading_days[1:]:
                run_daily(day, df_daily_olhc, df_trend)

            res_dir = 'result/' + str(startdate) + '_' + str(
                enddate) + '/' + code + '/switch_' + str(P.switch_ahead_days) + '_ahead/'
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)

            df_summary = summary(df_result, df_win)

            df_trend.to_csv(res_dir + 'trend.csv')
            df_result.to_csv(res_dir + 'result.csv')
            df_win.to_csv(res_dir + 'win ratio.csv')
            df_make_order_record.to_csv(res_dir + 'make_order_record.csv')
            df_summary.to_csv(res_dir + 'df_summary.csv')

            print(code, switch_ahead_day, 'finished')
