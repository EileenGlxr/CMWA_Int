# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:15:51 2020

@author: liuxr
"""

import cx_Oracle as oracle
from WindPy import *
import sys
import os
from sqlalchemy import *
from datetime import *
from dateutil.relativedelta import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from datetime import *
from dateutil.relativedelta import *
from dateutil.rrule import *
import time
import TSLPy3 as ts

#%%
def tsbytestostr(data):
    if (isinstance(data,(tuple)) or isinstance(data,(list))):
        lendata = len(data)
        ret = []
        for i in range(lendata):
            ret.append(tsbytestostr(data[i]))
    elif isinstance(data,(dict)):
        lendata = len(data)
        ret ={}
        for i in data:
            ret[tsbytestostr(i)] = tsbytestostr(data[i])
    elif isinstance(data,(bytes)):
        ret = data.decode('gbk')
    else:
        ret = data
    

    return ret


def getFuturesdata_daily(FuturesCode,startdate,enddate):
    thisinfo=[-1,[],None]
    tsl='''
    ThisBegT:=%dT;
    ThisEndT:=%dT;
    ThisTradeDate:=MarketTradeDayQk(ThisBegT,ThisEndT); //交易日期
    IndexFutureCode:= '%s';
  
    AllData:=Array();
    For DateI:=0 To Length(ThisTradeDate)-1 Do
    Begin
      ThisDate:=ThisTradeDate[DateI];
      TempContract:=UpperCase(GetFuturesID(IndexFutureCode,ThisDate));
      Tempbase:=array();
      for j:=0 to length(TempContract)-1 do
      Tempbase[TempContract[j]]:=spec(base(703018),TempContract[j]);
      ThisData:= select ['StockID'],DateToStr(['date']) as 'Date',TimeToStr(timeof(['date'])) as 'Time',
                    ['open'],['close'],['high'],['low'],['vol'],
                    ['sectional_cjbs'] As 'OI', //持仓量
                    Tempbase[['StockID']] As 'SettleDate'
                  from markettable datekey ThisDate to ThisDate+0.999999 of TempContract end;
      AllData&= ThisData;

      end;

    return AllData ;'''
    while thisinfo[0]!=0:
        thisinfo=tsbytestostr(ts.RemoteExecute(tsl% (startdate,enddate,FuturesCode),{}))
    thisdata=pd.DataFrame(thisinfo[1])
    thisdata['Date']=thisdata['Date'].map(lambda x: int(''.join(x.split('-'))))
    
    return thisdata

def getFuturesdata_30min(FuturesCode,startdate,enddate):
    
    thisinfo=[-1,[],None]
    tsl='''  
   ThisBegT:=%dT;
   ThisEndT:=%dT;
   IndexFutureCode:='%s';
   Ov:=BackUpSystemParameters2();
   SetSysParam(PN_Stock(),'SH000300');
   SetSysParam(PN_Precision(),4);

   SetSysParam(PN_Date(),ThisEndT);

   //现货指数代码
   IndexFutureList:=Array('IF','IH','IC');
   SpotIndexList:=Array('SH000300','SH000016','SH000905');
   CheckPos:=MFind(IndexFutureList,MCell=IndexFutureCode)[0][0];
   SpotIndexCode:=SpotIndexList[CheckPos];

   TempFirstDay:=StockFirstDay(IndexFutureCode+'00');  //期指首日上市日
   ThisBegT:=Max(ThisBegT,TempFirstDay); //保证在首日上市日之后
   setsysparam(pn_cycle(),cy_day());
   ThisTradeDate:=MarketTradeDayQk(ThisBegT,ThisEndT); //交易日期
   SetSysParam(PN_Cycle(),cy_30m());

   AllData:=Array();
   For DateI:=0 To Length(ThisTradeDate)-1 Do
   Begin
      ThisDate:=ThisTradeDate[DateI];
      TempContract:=UpperCase(GetFuturesID(IndexFutureCode,ThisDate));
      Tempbase:=array();
      for j:=0 to length(TempContract)-1 do
          Tempbase[TempContract[j]]:=spec(base(703018),TempContract[j]);
      //Tempbase:= select ['StockID'],['最后交易日'] from infotable 703 of  TempContract end;
      ThisData:= select ['StockID'],DateToStr(['date']) as 'Date',TimeToStr(timeof(['date'])) as 'Time',
                    ['open'],['close'],['high'],['low'],['vol'],
                    ['sectional_cjbs'] As 'OI', //持仓量
                    Tempbase[['StockID']] As 'SettleDate',
                    inttodate(Tempbase[['StockID']])-ThisDate As 'Maturity'
                  from markettable datekey ThisDate to ThisDate+0.999999 of TempContract end;
      AllData&= ThisData;
   End
   Return AllData;
    '''
    while thisinfo[0]!=0:
        thisinfo=tsbytestostr(ts.RemoteExecute(tsl% (startdate,enddate,FuturesCode),{}))
    thisdata=pd.DataFrame(thisinfo[1])
    thisdata['Date']=thisdata['Date'].map(lambda x: int(''.join(x.split('-'))))

    return thisdata

def getStock_5min_Data(stockID,startdate,enddate):
    thisinfo=[-1,[],None]
    #取后复权价
    tsl='''   
    Ov:=BackUpSystemParameters2();
   SetSysParam(PN_Stock(),'%s');
   SetSysParam(PN_Precision(),5);
   SetSysParam(PN_Rate(),1); 

   ThisBegT:=%dT;
   ThisEndT:=%dT;
   SetSysParam(PN_Date(),ThisEndT);
 

   StockID:='%s';

   ThisTradeDate:=MarketTradeDayQk(ThisBegT,ThisEndT); //交易日期
   SetSysParam(PN_Cycle(),cy_5m());
   
   AllData:=Array();
   For DateI:=0 To Length(ThisTradeDate)-1 Do
   Begin
      ThisDate:=ThisTradeDate[DateI];
      TempContract:=StockID;
      Tempbase:=array();
      for j:=0 to length(TempContract)-1 do
      ThisData:= select ['StockID'],DateToStr(['date']) as 'Date',TimeToStr(timeof(['date'])) as 'Time',
                    ['open'],['close'],['high'],['low'],['vol']
                  from markettable datekey ThisDate to ThisDate+0.999999 of TempContract end;
      AllData&= ThisData;

      end;
   Return AllData;
   '''
    while thisinfo[0]!=0:
        thisinfo=tsbytestostr(ts.RemoteExecute(tsl% (stockID,startdate,enddate,stockID),{}))
    thisdata=pd.DataFrame(thisinfo[1])
    thisdata['Date']=thisdata['Date'].map(lambda x: int(''.join(x.split('-'))))
    print('data_loaded from Tinysoft')
    return thisdata
    
        
#%%
if __name__=='__main__':

    ts.DefaultConnectAndLogin("test") 

    FuturesCodeList=['IF','IC','IH']
    
    startdate=20100101
    enddate=20200701
    
    for code in FuturesCodeList:
        
        dir_path = 'D:/data/30m_CTA/'
        
        df_daily_olhc_path = code+'_30m_'+str(startdate)+'_'+str(20200701)+'.csv'
        df_30m_path =code+ 'DailyTradeData_'+str(startdate)+'_'+str(20200701)+'.csv'
        if os.path.exists(dir_path+df_daily_olhc_path):
            df_daily_olhc = pd.read_csv(dir_path + df_daily_olhc_path)
            df_30m = pd.read_csv(dir_path + df_30m_path)
            print('data_read_from_files')
            
        else:
            df_daily_olhc = getFuturesdata_daily(code,startdate,enddate)
            df_30m = getFuturesdata_30min(code, startdate, enddate)
            print('data loaded from TynySoft')
            df_daily_olhc.to_csv(dir_path+df_daily_olhc_path)
            df_30m.to_csv(dir_path+df_30m_path)
    
