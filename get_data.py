# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
import numpy as np
import datetime
import time
import os
# pro = ts.pro_api('7ae709d0ba634357a957ca4798150dabecc3da0ac3b6829c4624765a')
ts.set_token('6144f3417ac5da6235442a7bafe9ba6931c3fba8dcdd8946d089f862')
# TODO 优化股票池
stock_pool = [
    '000001.SZ',
    '000002.SZ',
    '000063.SZ',
    '000166.SZ',
    '000333.SZ',
    '000538.SZ',
    '000568.SZ',
    '000651.SZ',
    '000661.SZ',
    '000725.SZ',
    '000776.SZ',
    '000858.SZ',
    '000876.SZ',
    '000895.SZ',
    '001979.SZ',
    '002024.SZ',
    '002027.SZ',
    '002142.SZ',
    '002304.SZ',
    '002352.SZ',
    '002415.SZ',
    '002475.SZ',
    '002594.SZ',
    '002607.SZ',
    '002714.SZ',
    '002736.SZ',
    '002938.SZ',
    '003816.SZ',
    '300015.SZ',
    '300059.SZ',
    '300122.SZ',
    '300498.SZ',
    '600000.SH',
    '600009.SH',
    '600015.SH',
    '600016.SH',
    '600018.SH',
    '600019.SH',
    '600028.SH',
    '600030.SH',
    '600031.SH',
    '600036.SH',
    '600048.SH',
    '600050.SH',
    '600104.SH',
    '600276.SH',
    '600309.SH',
    '600346.SH',
    '600406.SH',
    '600519.SH',
    '600585.SH',
    '600588.SH',
    '600606.SH',
    '600690.SH',
    '600745.SH',
    '600809.SH',
    '600837.SH',
    '600887.SH',
    '600900.SH',
    '600918.SH',
    '600999.SH',
    '601006.SH',
    '601012.SH',
    '601066.SH',
    '601088.SH',
    '601138.SH',
    '601166.SH',
    '601169.SH',
    '601186.SH',
    '601211.SH',
    '601225.SH',
    '601229.SH',
    '601288.SH',
    '601318.SH',
    '601319.SH',
    '601328.SH',
    '601336.SH',
    '601360.SH',
    '601390.SH',
    '601398.SH',
    '601601.SH',
    '601628.SH',
    '601658.SH',
    '601668.SH',
    '601688.SH',
    '601766.SH',
    '601800.SH',
    '601816.SH',
    '601818.SH',
    '601857.SH',
    '601888.SH',
    '601899.SH',
    '601933.SH',
    '601988.SH',
    '601989.SH',
    '601998.SH',
    '603160.SH',
    '603259.SH',
    '603288.SH',
    '603501.SH'
]


def get_data_ts(pool, start, end):
    for code in pool:
        df = ts.pro_bar(ts_code=code, adj='qfq', freq='M', start_date=start, end_date=end)
        pd.DataFrame.to_csv(df, './data_ts/'+code[:6]+'.csv')
        print(code)
        time.sleep(0.5)


# get_data_ts(stock_pool, '20191231', '20201231')
