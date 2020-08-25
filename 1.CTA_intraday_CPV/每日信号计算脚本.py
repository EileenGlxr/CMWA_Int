import pandas as pd
import numpy as np
import datetime as dt
import sys
import os

'''
calculate signal
'''


# 持仓量调整函数
def PV_calculation(data, open_data):
    '''
    :param data: dataframe,某日某合约分钟交易数据
    :param open_data: series
    :return: float，相关系数
    '''
    if open_data.empty:
        open_data = open_data.append([{'OI': 0}], ignore_index=True)

    daily_vol = data.vol.sum()  # 当日总成交量
    daily_delta_OI = (data.OI.iloc[-1] - open_data['OI']).iloc[0]  # 当日持仓量的总变化量

    delta_OI = data.OI.diff(1)
    delta_OI.iloc[0] = data.OI.iloc[0] - open_data['OI'].iloc[0]  # 补第一分钟的delta OI

    data.loc[:, 'OI_T1'] = data.loc[:, 'vol'] / daily_vol * daily_delta_OI
    data.loc[:, 'OI_T0'] = (-1) * (delta_OI - data['OI_T1'])
    data.loc[:, 'adj_OI_delta'] = data.loc[:, 'OI_T0'] + data.loc[:, 'OI_T1']

    data.loc[:, 'adj_OI'] = 0  # adj_OI记录调整后的OI序列
    data['adj_OI'].iloc[0] = open_data['OI'].iloc[0] + data['adj_OI_delta'].iloc[0]

    for i in range(1, len(data)):
        data['adj_OI'].iloc[i] = data['adj_OI'].iloc[i - 1] + data['adj_OI_delta'].iloc[i]

    delta_P = data['close'].diff().dropna()
    delta_adj_OI = data['adj_OI'].diff().dropna()
    corr = delta_P.corr(delta_adj_OI)
    # print(data['OI'].iloc[-1])
    # print(data['adj_OI'].iloc[-1])
    # data.to_csv('adj_OI_example.csv')  # 记录一日的OI调整过程

    return corr


# PV转换为交易信号
def PV_to_signal(PV_series):
    global thresh
    return_s = pd.Series(index=PV_series.index)

    return_s.loc[PV_series > thresh] = 1
    return_s.loc[PV_series < (-1) * thresh] = -1

    return return_s


# 加载初始数据，计算PV值
def load_data(path, open_data_path, etf_daily_path):
    global params
    global startdate

    df = pd.read_csv(path)
    df = df.loc[df.Date >= startdate]
    df_open = pd.read_csv(open_data_path)

    df_etf_daily_data = pd.read_csv(etf_daily_path, index_col=['Date'])
    df_etf_daily_data = df_etf_daily_data.loc[df_etf_daily_data.index >= startdate]

    tradingdates = df.Date.unique().tolist()

    settledates = dict()
    df_unique = df.drop_duplicates(['StockID'])
    for ind in df_unique.index:
        settledates[df_unique['StockID'].loc[ind]] = df_unique['SettleDate'].loc[ind]

    df_PV = pd.DataFrame(index=tradingdates,
                         columns=['curr_mon_con', 'next_mon_con', 'curr_mon_PV', 'next_mon_PV'])

    for day in tradingdates:
        day_iid = tradingdates.index(day)
        last_trading_date = tradingdates[day_iid - 1]

        data = df.loc[df.Date == day]
        contracts = data.StockID.unique()
        contracts.sort()

        df_PV.loc[day]['curr_mon_con'] = contracts[0]
        df_PV.loc[day]['next_mon_con'] = contracts[1]

        # 获取昨日收盘价格
        open_data = df_open.loc[df_open.Date == day]
        # open_data=df_open.loc[df.Date==last_trading_date]
        # open_data=open_data.loc[open_data['Time']=='15:00:00']

        # 修正持仓量并计算PV
        df_PV['curr_mon_PV'].loc[day] = PV_calculation(data.loc[data.StockID == contracts[0]],
                                                       open_data[open_data.StockID == contracts[0]])
        df_PV['next_mon_PV'].loc[day] = PV_calculation(data.loc[data.StockID == contracts[1]],
                                                       open_data[open_data.StockID == contracts[1]])

    df_PV['curr_signal'] = PV_to_signal(df_PV['curr_mon_PV'])
    df_PV['next_signal'] = PV_to_signal(df_PV['next_mon_PV'])

    return df_PV, df_etf_daily_data, settledates


# 计算信号
def signal_calculation(df_PV, settledates):
    tradingdates = df_PV.index.tolist()

    df_signal = pd.DataFrame(index=tradingdates,
                             columns=['SettleDateFlag', 'SettleWeekFlag', 'curr_signal', 'next_signal'])

    # 标记交易日
    settledates_lst=list(settledates.values())
    settledates_lst.sort()
    settledateinclude = [x for x in list(settledates.values()) if x in df_signal.index]
    settledateinclude.sort()
    settledateinclude.append(settledates_lst[settledates_lst.index(settledateinclude[-1])+1])


    df_signal['SettleDateFlag'].loc[settledateinclude[:-1]] = 1

    # 标记交易周
    iid = 0
    if settledateinclude:
        settleweek = dt.datetime.strptime(str(settledateinclude[iid]), "%Y%m%d").strftime('%W')
        settleyear = dt.datetime.strptime(str(settledateinclude[iid]), "%Y%m%d").strftime('%Y')

        for day_iid in range(len(tradingdates)):
            day = tradingdates[day_iid]
            date = dt.datetime.strptime(str(day), "%Y%m%d")
            week = date.strftime('%W')
            #print('day:',str(day),' week', str(week),' next_settle_date:',settledateinclude[iid])
            if week == settleweek:  # 处于交易周
                df_signal['SettleWeekFlag'].loc[day] = 1

            # 进入到新的一个交易周
            if week > settleweek and date.strftime(
                    '%Y') == settleyear:# or (week < settleweek and date.strftime('%Y') > settleyear):
                iid += 1
                if iid < len(settledateinclude):
                    # print('next settleweek:' + str(settleweek))
                    settleweek = dt.datetime.strptime(str(settledateinclude[iid]), "%Y%m%d").strftime('%W')
                    settleyear = dt.datetime.strptime(str(settledateinclude[iid]), "%Y%m%d").strftime('%Y')
                else:
                    pass
        df_PV['SettleWeekFlag'] = df_signal['SettleWeekFlag']

    # 删除下一个交易日间隔三天的信号
    for day in df_signal.index[:-1]:
        iid = df_signal.index.tolist().index(day)
        nexttradingdate = df_signal.index[iid + 1]

        delta_day = dt.datetime.strptime(str(nexttradingdate), "%Y%m%d") - dt.datetime.strptime(str(day), "%Y%m%d")
        if delta_day.days >= 4:
            df_signal.drop(day, inplace=True)

    if settledateinclude:  # 计算信号数据期间有交易日， 删除交易日
        df_signal = df_signal.drop(df_signal[df_signal['SettleDateFlag'] == 1].index)
        # 剔除交易周当月和下月信号不一的信号
        settleweekdays = df_signal[df_signal['SettleWeekFlag'] == 1].index

        for day in settleweekdays:
            PV_row = df_PV.loc[day]
            if PV_row['curr_signal'] == PV_row['next_signal']:
                df_signal['curr_signal'].loc[day] = PV_row['curr_signal']
                df_signal['next_signal'].loc[day] = PV_row['curr_signal']
            else:
                df_signal.drop(day, inplace=True)

        # 剩下的日期逐个根据信号填写
        for day in df_signal.index:
            if day in settleweekdays:
                continue
            else:
                PV_row = df_PV.loc[day]
                df_signal['curr_signal'].loc[day] = PV_row['curr_signal']
                df_signal['next_signal'].loc[day] = PV_row['curr_signal']

    else:  # 计算信号数据期间无交易日，直接填写
        for day in df_signal.index:
            PV_row = df_PV.loc[day]
            df_signal['curr_signal'].loc[day] = PV_row['curr_signal']
            df_signal['next_signal'].loc[day] = PV_row['curr_signal']

    df_signal.drop(df_signal.loc[df_signal['curr_signal'].isna()].index, inplace=True)
    df_signal.drop(df_signal.loc[df_signal['curr_signal'] < 1].index, inplace=True)  # 只保留看多的信号
    df_signal.drop(['next_signal', 'SettleDateFlag', 'SettleWeekFlag'], axis=1, inplace=True)

    df_PV.to_csv('D:/data/Futures_CTA/daily_signal/mod_0819_df_PV.csv')

    return df_signal


def winning_ratio_calculation(df_signal, df_etf_daily_data):
    df_etf_daily_data['openR'] = df_etf_daily_data['open'].pct_change(1)
    df_etf_daily_data['signal'] = df_signal.reindex(df_etf_daily_data.index)
    df_etf_daily_data['signal'].fillna(0, inplace=True)
    df_etf_daily_data['return_get'] = df_etf_daily_data['signal'].shift(2)
    df_etf_daily_data['signal_ret'] = df_etf_daily_data['openR'] * df_etf_daily_data['return_get']
    df_etf_daily_data['win'] = df_etf_daily_data['signal_ret'] > 0

    winning_ratio = df_etf_daily_data['win'].sum() / df_etf_daily_data['signal'].sum()

    return winning_ratio


# 数据处理
def handle_data(path, open_data_path, etf_daily_path):  # 计算因子

    df_PV, df_etf_daily_data, settledates = load_data(path, open_data_path, etf_daily_path)
    df_signal = signal_calculation(df_PV, settledates)
    #signal_winning_ratio = winning_ratio_calculation(df_signal, df_etf_daily_data)

    return df_etf_daily_data, df_signal


if __name__ == "__main__":
    
    currentpath=os.path.dirname(__file__)
    sys.path.append(currentpath)

    code = 'IF'
    thresh = 0.01
    etf = 'SH510300'
    startdate = 20180801
    dir_path = 'data/'


    path = dir_path + code + "_201704_202008.csv"  # 分钟级别数据
    open_data_path = dir_path + code + "_OpenData_201704_202008.csv"  # 开盘集合竞价
    etf_daily_path = dir_path + etf + "_DailyTradeData.csv"  # etf每日交易数据

    # handle_data

    df_etf_daily_data, df_signal = handle_data(path, open_data_path, etf_daily_path)
    # df_etf_daily_data.to_csv('data/daily_signal/0803_etf_signal_calculation.csv')
    df_signal.to_csv('result/daily_signal/mod_0824_df_signal_thresh_' + str(thresh) + '.csv')
