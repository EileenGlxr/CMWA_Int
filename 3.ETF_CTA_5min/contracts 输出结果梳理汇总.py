# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:12:08 2020

@author: liuxr
"""

import pandas as pd
import os
import sys
currentpath=os.path.dirname(__file__)
sys.path.append(currentpath)


#ETFs
# 0817

dir_path='result/'
# =============================================================================
# ETFs=os.listdir(dir_path+'ETF_OnlyLong_Jump_maL_30_LongBars_2')
# ETFs.sort()
# ETFs = [x for x in ETFs if not "." in x]
# 
# =============================================================================
ETFs=['芯片ETF','TMTETF','创业板','军工龙头ETF','新能源车ETF']
ma_Lengths=[10,20,30]
longHoldBars=[2,3,4,5,6]
trading_methods=['OnlyLong_Jump','BiDirection_Jump']


df_summary_all=pd.DataFrame(index=['annual_return', 'annual_return_std', 'return_std_ratio',  'maxdd',
           'total_fee_ratio', 'total_trading_days', 'total_switching_times'])

tradingdays=pd.read_csv(dir_path+'ETF_BiDirection_Jump_maL_10_LongBars_2/TMTETF/result.csv',index_col=0).index

df_net_value_all=pd.DataFrame(index=tradingdays)
df_net_value_all.index=pd.to_datetime(df_net_value_all.index,format='%Y%m%d')
print(ETFs)
for LongBars in longHoldBars:
    for L in ma_Lengths:
        for method in trading_methods:
            for ETF_name in ETFs:
                res_dir = dir_path + 'ETF_'+method+'_maL_'+str(L)+'_LongBars_'+str(LongBars)+'/'+ETF_name+'/'
                if os.path.exists(res_dir):
                    
                    df_summary = pd.read_csv(res_dir+'df_summary.csv', index_col=0)
                    df_summary_all[ETF_name+'_maL_'+str(L)+'_'+method+'_longBars'+str(LongBars)] = df_summary.iloc[:, 0]
                else:
                    continue
                df_result= pd.read_csv(res_dir+'result.csv', index_col=0,engine='python')
                df_result.index=pd.to_datetime(df_result.index,format='%Y%m%d')
                df_result.dropna(inplace=True)
                df_result['net_value']=df_result['Value']/df_result['Value'].iloc[0]
                
                df_net_value_all[ETF_name+'_maL_'+str(L)+'_'+method+'_longBars'+str(LongBars)]=df_result['net_value']


# =============================================================================
# df_summary_all.index=['年化收益','年化波动','收益波动比','最大回撤','总费率','总交易天数','总换仓次数']
# df_summary_all=df_summary_all.T
# df_summary_all.to_csv(dir_path+'summary_ETFs_all.csv',encoding='utf-8-sig')
# =============================================================================
df_net_value_all.to_csv(dir_path+'net_value_stocks_all.csv',encoding='utf-8-sig')

