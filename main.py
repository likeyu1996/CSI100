#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 李珂宇
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# numpy完整print输出
np.set_printoptions(threshold=np.inf)
# pandas完整print输出
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


class RandCSI100:
    def __init__(self, size, steps):
        self.size = size
        self.steps = steps
        self.stock_pool = np.array([])
        self.portfolio = np.array([])
        self.portfolio_data = pd.DataFrame
        self.portfolio_close = np.array([])
        self.simple_r_list = np.array([])
        self.log_r_list = np.array([])
        self.r_result_df = pd.DataFrame

    def load_stock_pool(self):
        stock_pool_cache = [
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
        # 清洗，剔除窗口期小于一年的数据
        stock_pool_dict = {i: pd.read_csv('./data_ts/'+stock_pool_cache[i][:6]+'.csv', na_values=np.nan, encoding='gb18030').loc[:, 'close'].to_numpy() for i in range(len(stock_pool_cache))}
        ii_list = []
        for key, value in stock_pool_dict.items():
            if len(value) < 13:
                print('股票{0}仅有{1}个数据，剔除'.format(stock_pool_cache[key][:6], len(value)))
                ii_list.append(key)
        stock_pool = np.delete(np.array(stock_pool_cache), ii_list)
        print('清洗结束，共剔除{0}个数据'.format(len(ii_list)))
        self.stock_pool = stock_pool

    def initialize_portfolio(self):
        portfolio_cache = np.random.choice(len(self.stock_pool), size=self.size, replace=False)
        self.portfolio = portfolio_cache
        portfolio_dict = {self.stock_pool[i][:6]: pd.read_csv('./data_ts/'+self.stock_pool[i][:6]+'.csv', na_values=np.nan, encoding='gb18030').loc[:, 'close'].to_numpy() for i in self.portfolio}
        '''
        portfolio_close_cache = [pd.read_csv('./data_ts/'+self.stock_pool[i][:6]+'.csv', na_values=np.nan, encoding='gb18030').loc[:, 'close'].to_numpy() for i in self.portfolio]
        for i in self.portfolio:
            data_raw = pd.read_csv('./data_ts/'+self.stock_pool[i][:6]+'.csv', na_values=np.nan, encoding='gb18030')
            stock_close = data_raw.loc[:, 'close'].to_numpy()
            portfolio_close_cache.append(stock_close)
            portfolio_dict[self.stock_pool[i][:6]] = stock_close
        
        portfolio_ewi_close = sum(portfolio_close_cache)
        portfolio_dict['EWI_sum'] = portfolio_ewi_close
        portfolio_dict['trade_date'] = data_raw.loc[:, 'trade_date'].to_numpy()
        '''
        # print(portfolio_dict.values())
        portfolio_dict['EWI_sum'] = sum(portfolio_dict.values())
        portfolio_dict['trade_date'] = pd.read_csv('./data_ts/'+self.stock_pool[0][:6]+'.csv', na_values=np.nan, encoding='gb18030').loc[:, 'trade_date'].to_numpy()
        portfolio_data = pd.DataFrame(portfolio_dict)
        cols = portfolio_data.columns.to_list()
        # 这里必须用list，用ndarray就不是这个效果了
        cols = cols[-2:-1]+cols[:-2]+cols[-1:]
        cols = cols[-1:]+cols[:-1]
        portfolio_data = portfolio_data[cols]
        self.portfolio_data = portfolio_data
        portfolio_data.to_csv('./result/portfolio.csv')

    def set_return_list(self):
        self.portfolio_close = self.portfolio_data.loc[:, 'EWI_sum'].to_numpy()
        # print(portfolio_close)
        self.simple_r_list = np.array([(self.portfolio_close[i]/self.portfolio_close[i+1]-1)
                                       for i in range(len(self.portfolio_close)-1)])
        self.log_r_list = np.array([np.log(self.portfolio_close[i]/self.portfolio_close[i+1])
                                    for i in range(len(self.portfolio_close)-1)])

    def calculator(self):
        self.load_stock_pool()
        simple_cache = {}
        log_cache = {}
        max_size = self.size
        for size in range(2, max_size+1):
            self.size = size
            simple_r_dict = {}
            log_r_dict = {}
            simple_r_result_dict = {}
            log_r_result_dict = {}
            for step in range(self.steps):
                self.initialize_portfolio()
                self.set_return_list()
                simple_r_dict[str(step+1)] = self.simple_r_list
                log_r_dict[str(step+1)] = self.log_r_list
                simple_r_result_dict[str(step+1)] = np.array([(self.portfolio_close[0]/self.portfolio_close[-1])**(1.0/12.0)-1,
                                                              np.var(self.simple_r_list)])
                log_r_result_dict[str(step+1)] = np.array([np.log(self.portfolio_close[0]/self.portfolio_close[-1])/12.0,
                                                           np.var(self.log_r_list)])
            simple_cache[str(size)] = sum(simple_r_result_dict.values())/self.steps
            log_cache[str(size)] = sum(log_r_result_dict.values())/self.steps
            # log_cache[str(size)] = np.mean(log_r_result_dict.values())
            simple_r_df = pd.DataFrame(simple_r_dict)
            log_r_df = pd.DataFrame(log_r_dict)
            simple_r_df.to_csv('./result/simple_r/'+str(size)+'.csv')
            log_r_df.to_csv('./result/log_r/'+str(size)+'.csv')
        simple_r_result_df = pd.DataFrame(simple_cache, index=['simple_mean', 'simple_var'])
        log_r_result_df = pd.DataFrame(log_cache, index=['log_mean', 'log_var'])
        r_result_df = pd.concat([simple_r_result_df, log_r_result_df], axis=0, join='outer').T.reset_index()
        r_result_df.rename(columns={'index': 'size'}, inplace=True)
        r_result_df.to_csv('./result/r_result.csv')
        self.r_result_df = r_result_df

    def draw_chart(self):
        data = self.r_result_df
        for i in range(1, data.columns.size):
            sns.set(rc={'figure.figsize': (16, 9)})
            sns.lineplot(x='size', y=data.columns[i], data=data, marker='*')
            plt.xlabel('Size')
            plt.ylabel('Value')
            plt.title(data.columns[i])
            bonus_cache = np.mean(data.loc[:, data.columns[i]].to_numpy())/100.0
            for x_value, y_value in zip(data.loc[:, 'size'].to_numpy(), data.loc[:, data.columns[i]].to_numpy()):
                # the position of the data label relative to the data point can be adjusted by adding/subtracting a value from the x &/ y coordinates
                plt.text(x=x_value,  # x-coordinate position of data label
                         y=y_value+bonus_cache,  # y-coordinate position of data label, adjusted to be 150 below the data point
                         s='{:.8f}'.format(y_value),  # data label, formatted to ignore decimals
                         color='purple')  # set colour of line
            plt.savefig('./chart/' + str(i) + '_' + data.columns[i].replace(':', '_') + '.png')
            plt.close('all')

    def test(self):
        self.calculator()
        self.draw_chart()

    def test2(self):
        self.load_stock_pool()
        self.initialize_portfolio()
        self.set_return_list()
        print(self.simple_r_list)
        '''
        sns.distplot(self.simple_r_list)
        plt.show()
        '''
        print(np.var(self.simple_r_list))
        print(np.var(self.log_r_list))


if __name__ == '__main__':
    x = RandCSI100(size=98, steps=20)
    x.test()
