# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:38:58 2020

@author: liuxr

策略说明：
    日内仅进行一次交易
    手续费单边万2.5
    加滑点：单边万3.5
    测试不同ma长度以及longHoldBars

"""

import sys
import pandas as pd
import numpy as np
import datetime as dt
import sys
import os
from load_data import  getStock_5min_Data


# 全局变量记录各种参数
class Params():
    def __init__(self, code,jump,ma_length,method,LongBars):
        if  'Jump' in method:
            self.fee=0.00025+jump
        else:
            self.fee=0.00025
            
        self.method=method
        self.maLength = ma_length
        self.longHoldBars = LongBars
        self.shortHoldBars = 1
        self.trail_start = 2
        self.trail_stop = 60
        self.ATRK = 3
        self.exitATRK = 2
        self.entryLots = 1

        self.tradeEndT = '14:00:00'
        self.closePosT = '14:55:00'

        self.initial_amount = None  # float,初始资金
        self.amount = None  # float,记录累计收益

        self.trading_days = None  # list

        self.open_day = None
        self.position = None  # str, 'long' or 'short'
        self.contract = None  # str, 持有的合约
        self.contract_num = 1  # int,持有的手数。本策略中始终为1
        self.trade_win = None
        self.daily_win = None
        self.netvalue = None  # float，每日收盘计算净值
        self.price = None
        
        self.mul = 100  # 一手100

        self.entrybar_iid = None
        self.trade_data = None

        self.contract_considered =code

    def atr_band_calculation(self, df):
        df.insert(len(df.columns), 'MA', None)
        df.insert(len(df.columns), 'ATR', None)
        df.insert(len(df.columns), 'upper_strike', None)
        df.insert(len(df.columns), 'lower_strike', None)
        df.insert(len(df.columns), 'upper_band', None)
        df.insert(len(df.columns), 'lower_band', None)
        df.insert(len(df.columns),'trade_signal',0)

        col_n = ['high', 'close', 'low']
        hlc = pd.DataFrame(df, columns=col_n).mean(axis=1)
        df['MA'] = hlc.rolling(self.maLength).mean()

        TR = pd.concat(
            [df['high'] - df['low'], abs(df['close'].shift(1) - df['high']), abs(df['close'].shift(1) - df['low'])],
            axis=1)
        df['ATR'] = TR.max(axis=1).rolling(14).mean()

        def strike_count(df, direction):
            count_series = pd.Series(index=df.index)
            count = 0
            if direction == 'upper':
                for i in df.index:
                    row = df.loc[i]
                    if row['low'] >= row['MA']:
                        count += 1
                    else:
                        count = 0
                    count_series.loc[i] = count

            if direction == 'lower':
                for i in df.index:
                    row = df.loc[i]
                    if row['high'] <= row['MA']:
                        count += 1
                    else:
                        count = 0
                    count_series.loc[i] = count
            return count_series

        df['upper_strike'] = strike_count(df, 'upper')
        df['lower_strike'] = strike_count(df, 'lower')

        for i in df.index:
            row = df.iloc[i]
            iid= df.index.tolist().index(i)
            if row['upper_strike']>=P.longHoldBars:
                df_used = df.iloc[iid - (int(row['upper_strike'] - 1)):iid + 1]
                df.loc[i, 'upper_band'] = max(df_used['high'])
            if row['lower_strike']>=P.shortHoldBars:
                df_used = df.iloc[iid - (int(row['lower_strike'] - 1)):iid + 1]
                df.loc[i, 'lower_band'] = min(df_used['low'])

        df['upper_band'].fillna(99999, inplace=True)
        df['lower_band'].fillna(0, inplace=True)

        df.reset_index(drop=True, inplace=True)

        self.trade_data = df

    def update_tradingdays(self, tradingdays):
        self.trading_days = tradingdays


# 下单函数
def make_order(day, time, contract, signal, price, entrybar_iid=None):
    global P

    global df_win
    global df_make_order_record

    if P.initial_amount == None:
        P.initial_amount = price * P.mul
        P.amount = P.initial_amount

    if signal == 'close_pos':  # 只有平仓的时候更新amount
# =============================================================================
#         if day == P.open_day:
#             fee_pct = P.fee * 15
#         else:
#             fee_pct = P.fee
# =============================================================================
        fee_pct=P.fee 
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


def daily_close_pos(day):
    global P
    global df_result

    today_trade_data = P.trade_data.loc[P.trade_data['Date'] == day]
    close_row = today_trade_data.iloc[-2]

    if P.contract:
        close_price = close_row['open']
        make_order(day,close_row['Time'],P.contract, 'close_pos', close_price)


# 收盘前更新净值,尚未更新
def update_before_close(day):
    global P
    global df_result

    P.netvalue = P.amount

    row = pd.DataFrame([[P.contract_considered, P.contract_num, P.netvalue]],
                       index=[day],
                       columns=['ETFID', 'ContractNum', 'Value'])

    df_result = pd.concat([df_result, row], axis=0)

def make_transactions_bidirect(bars,this_row,day,time,iid):
    global P
    # 向上突破upper_band
    if P.position != 'long' and bars['high'].iloc[iid] > bars['upper_band'].iloc[iid - 1] and bars['high'].iloc[
        iid - 1] <= bars['upper_band'].iloc[iid - 1]:
        if P.position == 'short':
            make_order(day, time, P.contract, 'close_pos',
                        max(bars['open'].iloc[iid], bars['upper_band'].iloc[iid - 1]), iid)
# =============================================================================
#             make_order(day, time, P.contract, 'close_pos',bars['open'].iloc[iid+1], iid)
# =============================================================================

        make_order(day, time, P.contract_considered, 'long',
                    max(bars['open'].iloc[iid], bars['upper_band'].iloc[iid - 1]), iid)
# =============================================================================
#         make_order(day, time, P.contract_considered, 'long',bars['open'].iloc[iid + 1], iid)
# =============================================================================
        P.trade_data.loc[this_row.index,'trade_signal']=1

    # 向下突破lower_band
    if P.position != 'short' and bars['low'].iloc[iid] < bars['lower_band'].iloc[iid - 1] and bars['low'].iloc[
        iid - 1] >= bars['lower_band'].iloc[iid - 1]:


        if P.position == 'long':
            make_order(day, time, P.contract, 'close_pos',
                       min(bars['open'].iloc[iid], bars['lower_band'].iloc[iid - 1]), iid)
# =============================================================================
#             make_order(day, time, P.contract, 'close_pos',bars['open'].iloc[iid + 1], iid)
# =============================================================================

        make_order(day, time, P.contract_considered, 'short', min(bars['open'].iloc[iid], bars['lower_band'].iloc[iid - 1]), iid)
# =============================================================================
#         make_order(day, time, P.contract_considered, 'short', bars['open'].iloc[iid+1], iid)
# =============================================================================
        P.trade_data.loc[this_row.index,'trade_signal']=-1
        
def make_transactions_onlylong(bars,this_row,day,time,iid):
    global P
    # 向上突破upper_band
    if P.position != 'long' and bars['high'].iloc[iid] > bars['upper_band'].iloc[iid - 1] and bars['high'].iloc[
        iid - 1] <= bars['upper_band'].iloc[iid - 1]:
        if P.position == 'short':
            make_order(day, time, P.contract, 'close_pos',
                        max(bars['open'].iloc[iid], bars['upper_band'].iloc[iid - 1]), iid)
# =============================================================================
#             make_order(day, time, P.contract, 'close_pos',bars['open'].iloc[iid+1], iid)
# =============================================================================

        make_order(day, time, P.contract_considered, 'long',
                    max(bars['open'].iloc[iid], bars['upper_band'].iloc[iid - 1]), iid)
# =============================================================================
#         make_order(day, time, P.contract_considered, 'long',bars['open'].iloc[iid + 1], iid)
# =============================================================================
        P.trade_data.loc[this_row.index,'trade_signal']=1

    # 向下突破lower_band
# =============================================================================
#     if P.position != 'short' and bars['low'].iloc[iid] < bars['lower_band'].iloc[iid - 1] and bars['low'].iloc[
#         iid - 1] >= bars['lower_band'].iloc[iid - 1]:
# 
# 
#         if P.position == 'long':
#             make_order(day, time, P.contract, 'close_pos',
#                        min(bars['open'].iloc[iid], bars['lower_band'].iloc[iid - 1]), iid)
# # =============================================================================
# #             make_order(day, time, P.contract, 'close_pos',bars['open'].iloc[iid + 1], iid)
# # =============================================================================
# 
#         make_order(day, time, P.contract_considered, 'short', min(bars['open'].iloc[iid], bars['lower_band'].iloc[iid - 1]), iid)
# # =============================================================================
# #         make_order(day, time, P.contract_considered, 'short', bars['open'].iloc[iid+1], iid)
# # =============================================================================
#         P.trade_data.loc[this_row.index,'trade_signal']=-1
# =============================================================================


def stoploss(bars,this_row,day,time,iid):
    global P
    
    # 非空仓止损止盈
    if P.contract and ~this_row['ATR'].isna().iloc[0]:

        bars_since_entry = iid - P.entrybar_iid

        temp_high = bars['high'].iloc[P.entrybar_iid:iid].max()
        temp_low = bars['low'].iloc[P.entrybar_iid:iid].min()

        if P.position == 'long' and bars_since_entry > 0:
            if temp_high > P.price * (1 + P.trail_start * 0.01):  # 止盈
                stop_line = temp_high - (temp_high - P.price) * P.trail_stop * 0.01
                if bars['low'].iloc[iid] < stop_line:
                    make_order(day, time, P.contract, 'close_pos', min(stop_line, bars['open'].iloc[iid]))

            elif temp_high <= P.price * (1 + P.trail_start * 0.01):  # 止损
                stop_line = P.price - bars['ATR'].iloc[iid - 1] * P.ATRK
                if bars['low'].iloc[iid] < stop_line:
                    make_order(day, time, P.contract, 'close_pos', min(stop_line, bars['open'].iloc[iid]))

        if P.position == 'short' and bars_since_entry > 0:
            if temp_low < P.price * (1 - P.trail_start * 0.01):  # 止盈
                stop_line = temp_low + (P.price - temp_low) * P.trail_stop * 0.01
                if bars['high'].iloc[iid] > stop_line:
                    make_order(day, time, P.contract, 'close_pos', max(stop_line, bars['open'].iloc[iid]))

            elif temp_low >= P.price * (1 - P.trail_start * 0.01):  # 止损
                stop_line = P.price + bars['ATR'].iloc[iid - 1] * P.ATRK
                if bars['high'].iloc[iid] > stop_line:
                    make_order(day, time, P.contract, 'close_pos', max(stop_line, bars['open'].iloc[iid]))


# 日内每个bar进行监控
def intraday_track(day):
    global P
    global df_make_order_record

    bars = P.trade_data.loc[P.trade_data['Date'] == day]
    tradetime = bars.loc[bars['Date'] == day]['Time'].unique()
    tradetime.sort()

    # 当日进行交易,9:30-14:00
    for time in tradetime[1:42]:
        this_row = bars.loc[(bars['Date'] == day) & (bars['Time'] == time)]
        iid = bars.index.tolist().index(this_row.index)  # 在当天bars中的iif
        #last_row = bars.iloc[iid - 1]

        if this_row.index < P.maLength:
            continue
        
        if df_make_order_record.empty:
            if 'OnlyLong' in P.method:
                make_transactions_onlylong(bars,this_row,day,time,iid)
            if 'BiDirection' in P.method:
                make_transactions_bidirect(bars, this_row, day, time, iid)
        else:
            last_trade_day=df_make_order_record.iloc[-1].Date
            if last_trade_day!=day:
                if 'OnlyLong' in P.method:
                    make_transactions_onlylong(bars,this_row,day,time,iid)
                if 'BiDirection' in P.method:
                    make_transactions_bidirect(bars, this_row, day, time, iid)
                
    
        stoploss(bars,this_row,day,time,iid)
        
        

def run_daily(day):
    global P

    intraday_track(day)

    daily_close_pos(day)

    update_before_close(day)


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
    global df_make_order_record

    df_result_sum = df_result.dropna()
    df_result_sum['daily_return'] = df_result_sum['Value'].pct_change(1)  # series

    # 年化收益率
    df_result_sum['net_value_1'] = df_result_sum['Value'] / df_result_sum['Value'].iloc[0]
    tempinterval = dt.datetime.strptime(str(df_result_sum.index[-1]), "%Y%m%d") - dt.datetime.strptime(
        str(df_result_sum.index[0]),
        "%Y%m%d")
    intervaldays = tempinterval.days
    intervalyears = intervaldays / 365
    annual_return = np.exp(np.log(df_result_sum['net_value_1'].iloc[-1]) / intervalyears) - 1  # 年化收益率

    annual_return_std = np.std(df_result_sum['daily_return']) * np.sqrt(252)  # 波动率
    return_std_ratio = annual_return / annual_return_std  # float


    # 手续费占比
    total_fee = df_win['transaction_fee'].sum()
    total_fee_ratio = total_fee / (
            df_result_sum['Value'].iloc[-1] - df_result_sum['Value'].iloc[0] + total_fee)

    # 最大回撤
    maxdd = calcmaxdd(df_result_sum['Value'])

    # 交易天数
    total_trading_days = len(df_result_sum.dropna())

    # 换仓次数
    total_switching_times = len(df_win)

    df_summary = pd.DataFrame(
        [annual_return, annual_return_std, return_std_ratio, maxdd, total_fee_ratio,
         total_trading_days, total_switching_times],
        index=['annual_return', 'annual_return_std', 'return_std_ratio', 'maxdd',
               'total_fee_ratio', 'total_trading_days', 'total_switching_times'])

    return df_summary


# %%

if __name__ == '__main__':
    currentpath=os.path.dirname(__file__)
    sys.path.append(currentpath)

    startdate = 20180801
    enddate = 20200810

    df_ETFs=pd.read_csv('original_data/ETFs.csv',index_col=0,engine='python')
    ma_Lengths=[10,20,30] # ma窗口产嘀咕
    trading_methods=['BiDirection_Jump','OnlyLong_Jump'] # 多空+滑点，仅作多头+滑点
    longHoldBars=[2,3,4,5,6] # 持续多少个bar在ma上时才做多。
    selected_ETFs=['芯片ETF','TMTETF','创业板','军工龙头ETF','新能源车ETF']
    for LongBars in longHoldBars:
        for L in ma_Lengths:
            for method in trading_methods:
                #for ind in df_ETFs.index:
                for ETF in selected_ETFs:
                    ind=df_ETFs['name'].tolist().index(ETF)+1
                    row=df_ETFs.loc[ind]
                    ETF_name=row['name']
                    ETF_jump=row['jump']

                    full_code=row['exchange']+str(row['code'])
                    res_dir = 'result/ETF_'+method+'_maL_'+str(L)+'_LongBars_'+str(LongBars)+'/'+ETF_name+'/'

                    if os.path.exists(res_dir):
                        print(ETF_name,'exists')
                        continue

                    df_5min_path = 'original_data/5m_'+ETF_name+'_CTA.csv'
                    if os.path.exists(df_5min_path):
                        df_5min = pd.read_csv( df_5min_path)
                        print('data_read_from_files')
                    else:
                        df_5min=getStock_5min_Data(full_code,startdate,enddate)
                        df_5min.to_csv(df_5min_path)
                        # print('data loaded from TinySoft')
                        
                    # 20180801至今
                    df_5min = df_5min.loc[df_5min['Date'] >= startdate]
                
                    P = Params(full_code,ETF_jump,L,method,LongBars)
                    P.update_tradingdays(list(df_5min.Date.unique()))
                    P.atr_band_calculation(df_5min)
                
                    df_result = pd.DataFrame(columns=['ETFID', 'ContractNum', 'Value'])
                    df_win = pd.DataFrame(columns=['Win', 'transaction_fee'])
                    df_make_order_record = pd.DataFrame(
                        columns=['Date', 'Time', 'FutureID', 'Action', 'Price', 'Profit', 'Amount'])
                
                    for day in P.trading_days:
                        run_daily(day)
                
                    if not os.path.exists(res_dir):
                        os.makedirs(res_dir)
                
                    df_summary = summary(df_result, df_win)
                
                    P.trade_data.to_csv(res_dir+'df_5min_signal.csv')
                    df_result.to_csv(res_dir + 'result.csv')
                    df_win.to_csv(res_dir + 'win ratio.csv')
                    df_make_order_record.to_csv(res_dir + 'make_order_record.csv')
                    df_summary.to_csv(res_dir + 'df_summary.csv')
                
                    print('finished',ETF_name)
            
