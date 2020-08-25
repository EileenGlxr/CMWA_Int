import pandas as pd
import numpy as np
import datetime as dt
import sys
import os

'''
CTA_original_300ETF：
    单边手续费单边2.5%%
    剔除交易日信号,交割周信号当月与下月合约不同则提出
    没有信号 - 开盘平仓
    纯多头 

'''


# Account类，记录持有合约、开仓日期、买入价格、持仓方向
class Account():
    def __init__(self):
        global code

        self.open_price = None
        self.open_day = None
        self.fee = None

        self.position = None  # str, 'long' or 'short'
        self.contract = None  # str, 持有的合约
        self.contract_num = None  # 持有的手数。累计净值，第一次后会增加

    def buy(self, day, contract, price):  # 做多
        global df_make_order_record
        global params

        self.contract_num = params.amount / ((price * params.mul) * (1 + params.fee_pct))

        buy_fee = params.amount - price * params.mul * self.contract_num
        params.amount = params.amount - buy_fee

        self.open_day = day
        self.open_price = price
        self.contract = contract
        self.position = 'long'

        row = pd.DataFrame([[day, contract, self.contract_num, 'long', price, buy_fee, None, params.amount]],
                           index=[day],
                           columns=['Date', 'FutureID', 'ContractNum', 'Action', 'Price', 'Fee', 'Profit', 'Amount'])
        df_make_order_record = pd.concat([df_make_order_record, row], axis=0)

    def sell(self, day, price):  # 平多
        global df_make_order_record
        global df_win

        sell_fee = params.fee_pct * (params.mul * price * self.contract_num)
        sell_and_get = self.contract_num * price * params.mul - sell_fee

        delta_amount = sell_and_get - params.amount
        params.amount = sell_and_get

        sell_contract = self.contract
        sell_contract_num = self.contract_num

        self.fee = None
        self.contract = None
        self.position = None
        self.contarct_num = None
        params.trade_win = True if price > self.open_price else False
        self.open_day = None
        self.open_price = None

        row = pd.DataFrame(
            [[day, sell_contract, sell_contract_num, 'close_pos', price, sell_fee, delta_amount, params.amount]],
            index=[day],
            columns=['Date', 'FutureID', 'ContractNum', 'Action', 'Price', 'Fee', 'Profit', 'Amount'])
        df_make_order_record = pd.concat([df_make_order_record, row], axis=0)

        row = pd.DataFrame([[params.trade_win]], index=[day], columns=['Win'])
        df_win = pd.concat([df_win, row], axis=0)

    def close_position(self, day, price):
        if self.position == 'long':
            self.sell(day, price)
        else:
            print('Error！')

    # 止损时锁仓账户调用
    def lock(self, day, acct2, price):
        if acct2.position == 'long':
            self.sell(day, price)
        else:
            print('The other acct is empty!')


# 记录策略参数
class Params():  # 参数
    def __init__(self, thresh, fund):
        global code
        self.thresh = thresh

        self.initial_amount = 100000  # float,初始资金
        self.amount = 100000  # float,记录累计收益

        self.mul = 100

        self.tradingdates = None  # df.index,期间所有交易日
        self.settledates = None  # dict，合约及对应交割日

        self.trade_win = None  # bool，交易是否获胜
        self.daily_win = None  # bool
        self.netvalue = None  # float，每日收盘计算净值
        self.contract_considered = fund

        self.fee_pct = 0.00025  # 单边交易手续费,单边万2.5

        self.dict_olhc = dict()  # key：合约，value：合约的日度高收低开
        self.ATRK = 2  # ATR止损参数
        self.window = 7  # ATR止损参数

    # 填充dict_con
    def split_dataframe(self, df_olhc):
        contracts = df_olhc.StockID.unique().tolist()
        for con in contracts:
            self.dict_olhc[con] = df_olhc.loc[df_olhc['StockID'] == con]
            self.dict_olhc[con] = self.dict_olhc[con].set_index('Date')

    # 计算ATR
    def atr_calculation(self):
        for con, df in self.dict_olhc.items():
            df.insert(len(df.columns), 'ATR', None)

            TR = pd.concat(
                [df['high'] - df['low'], abs(df['close'].shift(1) - df['high']), abs(df['close'].shift(1) - df['low'])],
                axis=1)
            df['ATR'] = TR.max(axis=1).rolling(self.window).mean()

            # df.reset_index(drop=True, inplace=True)

    # 更新回测期间的交易日期
    def update_dates(self, df, df_olhc):
        index1 = df.Date.unique()
        index2 = df_olhc.Date.unique()
        union_ind = list(set(index1).intersection(set(index2)))
        union_ind.sort()
        self.tradingdates = union_ind

    # 更新期指合约对应的到期日
    def update_settledates(self, df):
        self.settledates = dict()
        df_unique = df.drop_duplicates(['StockID'])
        for ind in df_unique.index:
            self.settledates[df_unique['StockID'].loc[ind]] = df_unique['SettleDate'].loc[ind]


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
    # data.to_csv('data/PV/'+str(data['Date'].iloc[0])+'_adj_OI_example.csv')  # 记录一日的OI调整过程

    return corr


# PV转换为交易信号
def PV_to_signal(PV_series):
    global params
    return_s = pd.Series(index=PV_series.index)

    return_s.loc[PV_series > params.thresh] = 1
    return_s.loc[PV_series < (-1) * params.thresh] = -1

    return return_s


# 加载初始数据，计算PV值
def load_data(path, open_data_path, daily_data_path, startdate):
    global params
    df = pd.read_csv(path)
    df_open = pd.read_csv(open_data_path)
    df_olhc = pd.read_csv(daily_data_path)

    params.split_dataframe(df_olhc)
    params.atr_calculation()

    df = df.loc[df['Date'] >= startdate]

    params.update_dates(df, df_olhc)
    params.update_settledates(df)

    tradingdates = params.tradingdates

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
    # df_PV.to_csv('data/PV/df_PV.csv')
    return df_PV, df_olhc


# 计算信号
def signal_calculation(df_PV):
    global params

    df_signal = pd.DataFrame(index=params.tradingdates,
                             columns=['SettleDateFlag', 'SettleWeekFlag', 'curr_signal', 'next_signal'])

    # 标记交易日
    settledateinclude = [x for x in list(params.settledates.values()) if x in df_signal.index]
    settledateinclude.sort()

    df_signal['SettleDateFlag'].loc[settledateinclude] = 1

    # 标记交易日
    settledates_lst = list(params.settledates.values())
    settledates_lst.sort()
    settledateinclude = [x for x in list(params.settledates.values()) if x in df_signal.index]
    settledateinclude.sort()
    settledateinclude.append(settledates_lst[settledates_lst.index(settledateinclude[-1]) + 1])

    df_signal['SettleDateFlag'].loc[settledateinclude[:-1]] = 1

    # 标记交易周
    iid = 0
    if settledateinclude:
        settleweek = dt.datetime.strptime(str(settledateinclude[iid]), "%Y%m%d").strftime('%W')
        settleyear = dt.datetime.strptime(str(settledateinclude[iid]), "%Y%m%d").strftime('%Y')

        for day_iid in range(len(params.tradingdates)):
            day = params.tradingdates[day_iid]
            date = dt.datetime.strptime(str(day), "%Y%m%d")
            week = date.strftime('%W')
            #print('day:', str(day), ' week', str(week), ' next_settle_date:', settledateinclude[iid])
            if week == settleweek:  # 处于交易周
                df_signal['SettleWeekFlag'].loc[day] = 1

            # 进入到新的一个交易周
            if week > settleweek and date.strftime(
                    '%Y') == settleyear:  # or (week < settleweek and date.strftime('%Y') > settleyear):
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

    # 删除交易日
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

    df_signal.drop(df_signal.loc[df_signal['curr_signal'].isna()].index, inplace=True)
    df_signal.drop(df_signal.loc[df_signal['curr_signal'] < 1].index, inplace=True)  # 只保留看多的信号

    return df_PV, df_signal


# 收盘前调用，更新并记录净值
def update_portfolio_before_close(day, df_olhc):
    global params
    global acct
    global acct_lc
    global df_result

    if acct.contract == None:
        params.netvalue = params.amount
    else:
        # print(day,acct.contract)
        trade_data = df_olhc.loc[(df_olhc['Date'] == day) & (df_olhc['StockID'] == acct.contract)].iloc[0]

        if acct.position == 'short':
            profit = acct.contract_num * params.mul * (acct.open_price - trade_data['close'])  # 做空价格-收盘价
            params.netvalue = params.amount + profit

        elif acct.position == 'long':
            profit = acct.contract_num * params.mul * (trade_data['close'] - acct.open_price)  # 做空价格-收盘价
            params.netvalue = params.amount + profit

        if acct.position == 'long' and trade_data['close'] > trade_data['open']:
            params.daily_win = True
        elif acct.position == 'short' and trade_data['close'] < trade_data['open']:
            params.daily_win = True
        else:
            params.daily_win = False

    row = pd.DataFrame([[acct.contract, acct.contract_num, acct.position, params.netvalue, params.daily_win]],
                       index=[day],
                       columns=['FutureID', 'ContractNum', 'Position', 'Value', 'Win'])

    df_result = pd.concat([df_result, row], axis=0)


# 开盘前调用，确定当日交易合约及交易信号。
def run_before_open(day, df_signal, df_PV):
    global params

    # if params.contract_considered == None:
    #     params.contract_considered = df_PV['curr_mon_con'].loc[day]

    # 调整合约：交割周开始换为下月合约
    # stlmnt_date = params.settledates[params.contract_considered]
    # days_to_maturity = dt.datetime.strptime(str(stlmnt_date), '%Y%m%d') - dt.datetime.strptime(str(day), '%Y%m%d')
    # if days_to_maturity.days < 5:
    #     params.contract_considered = df_PV['next_mon_con'].loc[day]

    # 判断signal
    iid = df_PV.index.tolist().index(day)
    last_trading_day = df_PV.index[iid - 1]
    signal_exist_flag = last_trading_day in df_signal.index

    return signal_exist_flag


# 开盘时调用，进行当日交易
def open_run(day, signal_exist_flag, df_PV, df_olhc):
    global params
    global acct
    global acct_lc

    iid = df_PV.index.tolist().index(day)
    last_trading_day = df_PV.index[iid - 1]
    today_trade_data = df_olhc.loc[(df_olhc['Date'] == day)]  # 当天四个合约的交易数据

    # if acct_lc.contract:  # 平备用仓
    #     if acct_lc.contract != params.contract_considered:
    #         close_price = today_trade_data.loc[today_trade_data['StockID'] == acct_lc.contract, 'open'].iloc[0]
    #         acct_lc.close_position(day, close_price)

    # if acct.contract:  # 交割日开盘平仓
    #     if day == params.settledates[acct.contract]:
    #         close_price = today_trade_data['open'].loc[today_trade_data['StockID'] == acct.contract].iloc[0]
    #         acct.close_position(day, close_price)

    if not signal_exist_flag:  # 前一个交易日没有信号，开盘平仓
        if acct.contract:
            close_price = today_trade_data.loc[today_trade_data['StockID'] == acct.contract, 'open'].iloc[0]
            acct.close_position(day, close_price)

    else:  # 有信号
        if acct.contract:
            if acct.contract != params.contract_considered:  # 换仓
                close_price = today_trade_data.loc[today_trade_data['StockID'] == acct.contract, 'open'].iloc[0]
                acct.close_position(day, close_price)

                open_price = \
                    today_trade_data.loc[today_trade_data['StockID'] == params.contract_considered, 'open'].iloc[0]
                signal = df_signal['next_signal'].loc[last_trading_day]
                if signal == 1:
                    acct.buy(day, params.contract_considered, open_price)
                # elif signal == -1:
                #     acct.sell_short(day, params.contract_considered, open_price)
                else:
                    pass

            curr_next = 'curr_' if acct.contract == df_PV['curr_mon_con'].loc[day] else 'next_'
            signal = df_signal[curr_next + 'signal'].loc[last_trading_day]
            last_signal = 1 if acct.position == 'long' else -1
            if signal == last_signal:
                pass
            else:
                close_price = today_trade_data['open'].loc[today_trade_data['StockID'] == acct.contract].iloc[0]
                acct.close_position(day, close_price)

                open_price = \
                    today_trade_data['open'].loc[today_trade_data['StockID'] == params.contract_considered].iloc[0]
                if signal == 1:
                    acct.buy(day, params.contract_considered, open_price)
                # elif signal == -1:
                #     acct.sell_short(day, params.contract_considered, open_price)
                else:
                    pass
        else:
            curr_next = 'curr_' if acct.contract == df_PV['curr_mon_con'].loc[day] else 'next_'
            signal = df_signal[curr_next + 'signal'].loc[last_trading_day]
            open_price = \
                today_trade_data['open'].loc[today_trade_data['StockID'] == params.contract_considered].iloc[0]
            if signal == 1:
                acct.buy(day, params.contract_considered, open_price)
            # elif signal == -1:
            #     acct.sell_short(day, params.contract_considered, open_price)
            else:
                pass


# 止损函数，不调用
# def stop_loss(day, df_olhc):
#     global params
#     global acct
#     global acct_lc
#
#     if not acct.contract:
#         pass
#     else:
#         trade_data = df_olhc.loc[(df_olhc['Date'] == day) & (df_olhc['StockID'] == acct.contract)].iloc[0]
#         iid = params.dict_olhc[acct.contract].index.tolist().index(day)
#
#         if acct.position == 'long':
#             stop_line = acct.open_price - params.dict_olhc[acct.contract]['ATR'].iloc[iid - 1] * params.ATRK
#             if trade_data['low'] < stop_line:
#                 if day != acct.open_day:
#                     lock_price = min(stop_line, trade_data['open'])
#                     acct.close_position(day, lock_price)
#                 else:
#                     acct_lc.lock(day, acct, stop_line)
#
#         elif acct.position == 'short':
#             stop_line = acct.open_price + params.dict_olhc[acct.contract]['ATR'].iloc[iid - 1] * params.ATRK
#             if trade_data['high'] > stop_line:
#                 if day != acct.open_day:
#                     close_price = max(stop_line, trade_data['open'])
#                     acct.close_position(day, close_price)
#                 else:
#                     acct_lc.lock(day, acct, stop_line)


# 每日运行函数
def run_daily(day, df_signal, df_PV, df_olhc):  # 每日调用，开仓换仓操作
    global params
    signal_exist_flag = run_before_open(day, df_signal, df_PV)

    open_run(day, signal_exist_flag, df_PV, df_olhc)

    # stop_loss(day, df_olhc) #是否添加止损

    # 收盘 计算当日净值
    update_portfolio_before_close(day, df_olhc)


def iopv_filter(df_signal, iopv_path):
    df_iopv = pd.read_csv(iopv_path, index_col=0)
    df_signal['discount'] = df_iopv['DiscountRate'].reindex()
    df_signal['dis_flag'] = df_signal['discount'].map(lambda x: 2 * (x < 0) - 1)  # 折价为1，溢价为-1
    df_signal['save_flag'] = (df_signal['curr_signal'] == df_signal['dis_flag'])

    df_signal.drop(df_signal.loc[df_signal['save_flag'] == False].index, inplace=True)

    return df_signal


# 数据处理
def handle_data(path, open_data_path, daily_data_path, startdate):  # 计算因子

    df_PV, df_olhc = load_data(path, open_data_path, daily_data_path, startdate)
    df_PV, df_signal = signal_calculation(df_PV)
    #df_PV.to_csv('data/daily_signal/df_PV_0731.csv')

   # df_signal.to_csv('data/daily_signal/df_signal_0731_thresh_' + str(params.thresh) + '.csv')
    # df_signal=iopv_filter(df_signal,iopv_path)

    return df_PV, df_olhc, df_signal


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


# 策略表现计算
def summary(df_result, df_win):
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

    # 日度胜率
    daily_win_ratio = len(df_result_sum.loc[df_result_sum['Win'] == True]) / len(df_result_sum)  # float

    # 交易胜率
    trades_win_ratio = sum(df_result_sum['Value'].diff()>0) / len(df_signal)

    # 手续费占比
    total_fee = df_make_order_record['Fee'].sum()
    total_fee_ratio = total_fee / (
            df_result_sum['Value'].iloc[-1] - df_result_sum['Value'].iloc[0] + total_fee)

    # 最大回撤
    maxdd = calcmaxdd(df_result_sum['Value'])

    # 交易天数
    total_trading_days = len(df_result_sum.dropna())

    # 换仓次数
    total_switching_times = len(df_win)

    df_summary = pd.DataFrame(
        [annual_return, annual_return_std, return_std_ratio, daily_win_ratio, trades_win_ratio, maxdd, total_fee_ratio,
         total_trading_days, total_switching_times],
        index=['annual_return', 'annual_return_std', 'return_std_ratio', 'daily_win_ratio', 'trade_win_ratio', 'maxdd',
               'total_fee_ratio', 'total_trading_days', 'total_switching_times'])

    return df_summary


if __name__ == "__main__":
    currentpath=os.path.dirname(__file__)
    sys.path.append(currentpath)
    
    future_code_list = ['IF']  # , 'IC', 'IH']
    startdate = 20180801
    # signal_threshs = [0, 0.005, 0.01, 0.03, 0.05]
    signal_threshs=[0.01]
    FundIDs = ['SH510300']#, 'SH510330', 'SZ159919']


    for code in future_code_list:
        for signal_thresh in signal_threshs:
            for fund in FundIDs:
                path ='data/'+ code + "_201704_202008.csv"  # 分钟级别数据
                open_data_path = 'data/'+ code + "_OpenData_201704_202008.csv"  # 开盘集合竞价
                daily_data_path = 'data/' + fund + "_DailyTradeData.csv"  # 每天高低收开
                # iopv_path = "data/300ETF_IOPV_close.csv"  # 300ETF的iopv

                params = Params(signal_thresh, fund)
                acct = Account()

                df_result = pd.DataFrame(columns=['FutureID', 'ContractNum', 'Position', 'Value', 'Win'])
                df_win = pd.DataFrame(columns=['Win'])
                df_make_order_record = pd.DataFrame(
                    columns=['Date', 'FutureID', 'ContractNum', 'Action', 'Price', 'Profit', 'Amount'])

                # handle_data
                df_PV, df_olhc, df_signal = handle_data(path, open_data_path, daily_data_path, startdate)

                for day in params.tradingdates[1:]:
                    run_daily(day, df_signal, df_PV, df_olhc)

                # summary
                df_summary = summary(df_result, df_win)

                pre_path ='result/'+code+'_300ETF_FullPosition_' + fund + '/' + 'thresh_added_' + str(
                    params.thresh) + '_' + str(
                    startdate) + '/'
                if not os.path.exists(pre_path):
                    os.makedirs(pre_path)

                df_PV.to_csv(pre_path + 'df_PV_' + str(signal_thresh) + '.csv')
                df_signal.to_csv(pre_path + 'df_signal_' + str(signal_thresh) + '.csv')
                df_result.to_csv(pre_path + 'result_' + str(signal_thresh) + '.csv')
                df_win.to_csv(pre_path + 'win ratio_' + str(signal_thresh) + '.csv')
                df_make_order_record.to_csv(pre_path + 'make_order_record_' + str(signal_thresh) + '.csv')
                df_summary.to_csv(pre_path + 'df_summary_' + str(signal_thresh) + '.csv')

                del params
                print(code, signal_thresh, 'finished')
