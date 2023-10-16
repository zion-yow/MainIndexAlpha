import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
from tqdm import tqdm
import warnings


weights50 = pd.read_csv(r'DailyFactor\weights50.csv',index_col=0)
weights300 = pd.read_csv(r'DailyFactor\weights300.csv',index_col=0)
weights500 = pd.read_csv(r'DailyFactor\weights500.csv',index_col=0)
weights1000 = pd.read_csv(r'DailyFactor\weights1000.csv',index_col=0)


# 标准化
def _Standlize(Fdf):
    return (Fdf - Fdf.mean()) / Fdf.std()

# （读取） + shift + 标准化 + 映射到指数
def read_project(name:str,trd_dates, df=None, method='value')->pd.DataFrame:
    if df is None:
        # 因子shift 1
        F = pd.read_csv(r'DailyFactor' + '/' + name + '.csv',index_col=0).shift(1,axis=1)
        F = _Standlize(F)
    else:
        F = df.shift(1,axis=1)
        F = _Standlize(F)
        
    if method == 'value':    
        Index_factor_panel = pd.concat([
            (weights50 * F).sum().rename('50'),
            (weights300 * F).sum().rename('300'),
            (weights500 * F).sum().rename('500'),
            (weights1000 * F).sum().rename('1000')
        ],axis=1).T.loc[:,trd_dates]
        
    elif method == 'equal':
        Index_factor_panel = pd.concat([
            ((weights50/weights50) * F).sum().rename('50'),
            ((weights300/weights300) * F).sum().rename('300'),
            ((weights500/weights500) * F).sum().rename('500'),
            ((weights1000/weights1000) * F).sum().rename('1000')
        ],axis=1).T.loc[:,trd_dates]
    
    Z_panel = _Standlize(Index_factor_panel)
    return Z_panel

def read_index_factor(name:str,trd_dates, df=None)->pd.DataFrame:
    if df is None:
        # 因子shift 1
        F = pd.read_csv(r'NewFactor' + '/' + name + '.csv',index_col=0).shift(1,axis=1)
        F = _Standlize(F)
    else:
        F = df.shift(1,axis=1)
        F = _Standlize(F)
    
    Z_panel = _Standlize(F)
    Z_panel.index = Z_panel.index.map(lambda x:str(x))
    
    return Z_panel



# 最大回撤（基于收益率序列）
def cal_downdraw(arr):
    I = np.argmax(np.maximum.accumulate(arr.cumsum()) - arr.cumsum()) 
    J = np.argmax(arr.cumsum()[:I])
    return arr.cumsum()[J] - arr.cumsum()[I]


# 指数截面单因子回测
def single_Factor_backtest(
    Fdf:pd.DataFrame,
    Index_rets:pd.DataFrame,
    _side:float,
    fee:float=0.0005,
    name='XX'
):
    max_values = Fdf.max()
    min_values = Fdf.min()

    # Replace the cells with the max value as 1 and min value as -1
    Index_panel_max_1 = Fdf.eq(max_values)
    Index_panel_min_minus1 = Fdf.eq(min_values)

    # Assign 1 to the cells with max value and -1 to the cells with min value
    Index_panel_select = Fdf.copy()
    
    Index_panel_select[Index_panel_max_1] = _side * 1
    Index_panel_select[Index_panel_min_minus1] = _side * -1
    Index_panel_select[~(Index_panel_min_minus1 | Index_panel_max_1)] = np.nan

    # cost
    fee_series = ((Index_panel_select.fillna(0) - Index_panel_select.shift(1,axis=1).fillna(0)).abs()*fee).sum()
    
    print((Index_panel_select.fillna(0) - Index_panel_select.shift(1,axis=1).fillna(0)).abs().sum().sum())
    
    # draw
    fig, ax = plt.subplots()
    (((Index_panel_select * Index_rets).sum() - fee_series)).cumsum().plot(ax=ax, title=f'Factor {name}', figsize=(15,8), grid=True)
    (Index_rets.T.cumsum()).plot(ax=ax, label='refer Index', style='--', alpha=0.5, figsize=(15,8), grid=True)
    
    ret = ((Index_panel_select * Index_rets).sum() - fee_series)
    
    mdd = -cal_downdraw(ret.dropna())
    sharpe = (ret.dropna().sum())/len(ret.dropna())*250 /  ((ret.dropna().std()* 250**0.5))
    rankIC = Index_rets.rank(ascending=False).corrwith(Fdf.rank(ascending=False)).mean()
    rankIR = Index_rets.rank(ascending=False).corrwith(Fdf.rank(ascending=False)).mean() / Index_rets.rank(ascending=False).corrwith(Fdf.rank(ascending=False)).std()
    
    print(f'mdd: {mdd}','\n', f'sharpe: {sharpe}','\n', f'rankIC: {rankIC}','\n', f'rankIR: {rankIR}')
    
    return Index_panel_select, ret


# 叠加隔夜预测
def T_trade_ret(select, T_select, bar_rets, Index_rets):
    date_set = sorted(list(set(select.columns)&set(T_select.columns)))
    _filter = (select==1) & (select != select.shift(1,axis=1)) & (T_select.loc[:,date_set]!=select.loc[:,date_set])
    
    _t_rets = (select[_filter] * bar_rets.loc[:,date_set])
    t_rets = _t_rets.mask((pd.isna(_t_rets)),Index_rets.loc[:,date_set])

    return t_rets


#计算分组收益
def cal_rate_group(factor,rets,group_num=10,masks=[1]):
    #分组回测
    group_lst = [(i/group_num,(i+1)/group_num) for i in range(group_num)]
    
    if masks is not pd.DataFrame:
        pass        
    else:
        factor = factor * (masks/masks)
 
    code_lst = np.intersect1d(factor.index,rets.index)
    date_lst = np.intersect1d(factor.columns,rets.columns)
    
    #filter0 剔除t日收盘涨跌停 t+1日开盘涨跌停
    factor = factor.loc[code_lst,date_lst]
    
    rets = rets.loc[code_lst,date_lst]
    factor_ratio = factor.rank(method = 'first',pct = True)
    factor_group = [factor_ratio[(factor_ratio>down_line) & (factor_ratio<=up_line)] for down_line , up_line in group_lst]
    factor_group = [i/i for i in factor_group]
    rate_group = []
    
    for j, i in enumerate(factor_group):
        rate_group.append((rets*i).mean())
        print(j+1, i.count().mean())
        
    
    df = pd.DataFrame(rate_group,index = [('group' + str(i+1)) for i in range(group_num)])
    
    return df.T.iloc[:-1,:]