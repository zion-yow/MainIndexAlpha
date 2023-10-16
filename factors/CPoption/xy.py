import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import mat4py
from tqdm import tqdm
import warnings
# Suppress warnings`
warnings.filterwarnings("ignore")



# 个股行情
opens = pd.read_csv(r'E:\PyProject\factors\DailyFactor\bs_pre\opens.csv',index_col=0)
highs = pd.read_csv(r'E:\PyProject\factors\DailyFactor\bs_pre\highs.csv',index_col=0)
lows = pd.read_csv(r'E:\PyProject\factors\DailyFactor\bs_pre\lows.csv',index_col=0)
closes = pd.read_csv(r'E:\PyProject\factors\DailyFactor\bs_pre\closes.csv',index_col=0)
precloses = pd.read_csv(r'E:\PyProject\factors\DailyFactor\bs_pre\precloses.csv',index_col=0)
volumes = pd.read_csv(r'E:\PyProject\factors\DailyFactor\bs_pre\volumes.csv',index_col=0)
amounts = pd.read_csv(r'E:\PyProject\factors\DailyFactor\bs_pre\amounts.csv',index_col=0)
turns = pd.read_csv(r'E:\PyProject\factors\DailyFactor\bs_pre\turns.csv',index_col=0)
pctChgs = pd.read_csv(r'E:\PyProject\factors\DailyFactor\bs_pre\pctChgs.csv',index_col=0)
isSTs = pd.read_csv(r'E:\PyProject\factors\DailyFactor\bs_pre\isSTs.csv',index_col=0)

weights50 = pd.read_csv(r'E:\PyProject\factors\DailyFactor\weights50.csv',index_col=0)
weights300 = pd.read_csv(r'E:\PyProject\factors\DailyFactor\weights300.csv',index_col=0)
weights500 = pd.read_csv(r'E:\PyProject\factors\DailyFactor\weights500.csv',index_col=0)
weights1000 = pd.read_csv(r'E:\PyProject\factors\DailyFactor\weights1000.csv',index_col=0)
weights2000 = pd.read_csv(r'E:\PyProject\factors\DailyFactor\weights2000.csv',index_col=0)
weights100sz = pd.read_csv(r'E:\PyProject\factors\DailyFactor\weights100sz.csv',index_col=0)
weights50kc = pd.read_csv(r'E:\PyProject\factors\DailyFactor\weights50kc.csv',index_col=0)
rets = closes.pct_change(axis=1)

trd_dates = sorted(list(set(rets.columns[1:])&set(weights50.columns)))

# 指数最高价
index_high = pd.read_csv(r'E:\PyProject\index_high.csv',index_col=0).T
index_high = index_high.mask(index_high==0,np.nan)
index_high.index = ['50','300','500','1000']
index_high.columns = [str(int(i))[:4] + '-' + str(int(i))[4:6] + '-' + str(int(i))[6:] for i in index_high.columns]

# 指数最低价
index_low = pd.read_csv(r'E:\PyProject\index_low.csv',index_col=0).T
index_low = index_low.mask(index_low==0,np.nan)
index_low.index = ['50','300','500','1000']
index_low.columns = [str(int(i))[:4] + '-' + str(int(i))[4:6] + '-' + str(int(i))[6:] for i in index_low.columns]

# 指数成交量
index_volume = pd.read_csv(r'E:\PyProject\index_volume.csv',index_col=0).T
index_volume = index_volume.mask(index_volume==0,np.nan)
index_volume.index = ['50','300','500','1000']
index_volume.columns = [str(int(i))[:4] + '-' + str(int(i))[4:6] + '-' + str(int(i))[6:] for i in index_volume.columns]

# 指数换手率
index_tov = pd.read_csv(r'E:\PyProject\index_tov.csv',index_col=0).T
index_tov = index_tov.mask(index_tov==0,np.nan)
index_tov.index = ['50','300','500','1000']
index_tov.columns = [str(int(i))[:4] + '-' + str(int(i))[4:6] + '-' + str(int(i))[6:] for i in index_tov.columns]

# 指数cl-cl收益率
cl = pd.read_csv(r'E:\PyProject\index_close.csv',index_col=0).T
cl = cl.mask(cl==0,np.nan)
cl.index = ['50','300','500','1000','sz100','kc50','2000']
cl.columns = [str(int(i))[:4] + '-' + str(int(i))[4:6] + '-' + str(int(i))[6:] for i in cl.columns]
Index_rets = cl.pct_change(axis=1)
Index_rets = Index_rets.T.where(~np.isinf(Index_rets.T),np.nan).T.loc[:,:]

# 指数cl-op收益率
op = pd.read_csv(r'E:\PyProject\index_open.csv',index_col=0).T
op.index = ['50', '300', '500', '1000', 'sz100', 'kc50','2000']
op.columns = [str(int(i))[:4] + '-' + str(int(i))[4:6] + '-' + str(int(i))[6:] for i in op.columns]
pcl = pd.read_csv(r'E:\PyProject\index_preclose.csv',index_col=0).T
pcl.index = ['50', '300', '500', '1000', 'sz100', 'kc50','2000']
pcl.columns = [str(int(i))[:4] + '-' + str(int(i))[4:6] + '-' + str(int(i))[6:] for i in pcl.columns]
open_cl_rets = op /  pcl - 1

# 补分红
div_exd = pd.read_csv(r'E:\PyProject\index_div_exd.csv',index_col=0)
div_exd.index = ['50', '300', '500', '1000', 'sz100', 'kc50','2000']
bonus_op = (op + div_exd).dropna(axis=1)
bonus_open_cl_rets = bonus_op /  cl.shift(1,axis=1) - 1

# 除st
ifst = pd.read_csv(r'E:\PyProject\GeneralData\StocksPanelData\ifst.csv',index_col=0)
ifst = ifst.T.mask(ifst.T=='是',np.nan)
ifst = ifst.mask(ifst=='否',1)
ifst.columns = ifst.columns.map(lambda x: str(x)[:4] +'-'+ str(x)[4:6] + '-' + str(x)[6:])
ifst = ifst.astype('float32')
# 除次新

filter0 = (volumes.T.rolling(60).count()>0).T
_filter0 = filter0.mask(filter0,1)
_filter0 = _filter0.mask(pd.isna(_filter0),np.nan)

filter1 = ifst * _filter0


# 指数分钟行情
IndexMin50jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'IndexMin50.csv',index_col=0)
IndexMin300jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'IndexMin300.csv',index_col=0)
IndexMin500jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'IndexMin500.csv',index_col=0)
IndexMin1000jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'IndexMin1000.csv',index_col=0)
IndexMin100sz_jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'IndexMin100sz.csv',index_col=0)
IndexMin2000jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'IndexMin2000.csv',index_col=0)
IndexMin50kc_jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'IndexMin50kc.csv',index_col=0)

# ETF
ETFMin50jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'ETFMin50.csv',index_col=0)
ETFMin300jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'ETFMin300.csv',index_col=0)
ETFMin500jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'ETFMin500.csv',index_col=0)
ETFMin1000jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'ETFMin1000.csv',index_col=0)
ETFMin100sz_jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'ETFMin100sz.csv',index_col=0)
ETFMin2000jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'ETFMin2000.csv',index_col=0)
ETFMin50kc_jq = pd.read_csv(r'E:\PyProject\GeneralData\IndexMinKline' + '\\' + f'ETFMin50kc.csv',index_col=0)

# 期货
Future_Min50jq = pd.read_csv(r'\\share\FREE\qza\IndexMinKline' + '\\' + f'IH_Min.csv',index_col=0)
Future_Min300jq = pd.read_csv(r'\\share\FREE\qza\IndexMinKline' + '\\' + f'IF_Min.csv',index_col=0)
Future_Min500jq = pd.read_csv(r'\\share\FREE\qza\IndexMinKline' + '\\' + f'IC_Min.csv',index_col=0)
Future_Min1000jq = pd.read_csv(r'\\share\FREE\qza\IndexMinKline' + '\\' + f'IM_Min.csv',index_col=0)


# 收益率
def rolling_window(a, window):
    a = np.array(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def Index_twap_ret(data, close_win, open_win):
    twap_close7 = pd.Series(np.nanmean(rolling_window(data.close,240)[::240][:,-close_win:],axis=1),index=data.index.str[:10][::240])
    twap_open30 = pd.Series(np.nanmean(rolling_window(data.close,240)[::240][:,:open_win],axis=1),index=data.index.str[:10][::240])
    
    return twap_open30 / twap_close7.shift(1) - 1

def ETF_cl_ret(data):
    twap_close = pd.Series(np.nanmean(rolling_window(data.close,240)[::240][:,-1:],axis=1),index=data.index.str[:10][::240])
    
    return twap_close / twap_close.shift(1) - 1

index_twap_ret10_30 = pd.concat([
    Index_twap_ret(IndexMin50jq, 10, 30).rename('50'),
    Index_twap_ret(IndexMin300jq, 10, 30).rename('300'),
    Index_twap_ret(IndexMin500jq, 10, 30).rename('500'),
    Index_twap_ret(IndexMin1000jq, 10, 30).rename('1000'),
    Index_twap_ret(IndexMin100sz_jq, 10, 30).rename('100sz'),
    Index_twap_ret(IndexMin2000jq, 10, 30).rename('2000'),
    Index_twap_ret(IndexMin50kc_jq, 10, 30).rename('50kc'),
],axis=1)

index_twap_ret1_1 = pd.concat([
    Index_twap_ret(IndexMin50jq, 1, 1).rename('50'),
    Index_twap_ret(IndexMin300jq, 1, 1).rename('300'),
    Index_twap_ret(IndexMin500jq, 1, 1).rename('500'),
    Index_twap_ret(IndexMin1000jq, 1, 1).rename('1000'),
    Index_twap_ret(IndexMin100sz_jq, 1, 1).rename('100sz'),
    Index_twap_ret(IndexMin2000jq, 1, 1).rename('2000'),
    Index_twap_ret(IndexMin50kc_jq, 1, 1).rename('50kc'),
],axis=1)

ETF_twap_ret10_30 = pd.concat([
    Index_twap_ret(ETFMin50jq, 10, 30).rename('50'),
    Index_twap_ret(ETFMin300jq, 10, 30).rename('300'),
    Index_twap_ret(ETFMin500jq, 10, 30).rename('500'),
    Index_twap_ret(ETFMin1000jq, 10, 30).rename('1000'),
    Index_twap_ret(ETFMin100sz_jq, 10, 30).rename('100sz'),
    Index_twap_ret(ETFMin2000jq, 10, 30).rename('2000'),
    Index_twap_ret(ETFMin50kc_jq, 10, 30).rename('50kc'),
],axis=1)

ETF_rets = pd.concat([
    ETF_cl_ret(ETFMin50jq).rename('50'),
    ETF_cl_ret(ETFMin300jq).rename('300'),
    ETF_cl_ret(ETFMin500jq).rename('500'),
    ETF_cl_ret(ETFMin1000jq).rename('1000'),
    ETF_cl_ret(ETFMin100sz_jq).rename('100sz'),
    ETF_cl_ret(ETFMin2000jq).rename('2000'),
    ETF_cl_ret(ETFMin50kc_jq).rename('50kc'),
],axis=1)





# 回测与评估
def _Standlize(Fdf):
    return (Fdf - Fdf.mean()) / Fdf.std()

def cal_downdraw(arr):
    I = np.argmax(np.maximum.accumulate(arr.cumsum()) - arr.cumsum()) 
    J = np.argmax(arr.cumsum()[:I])
    return arr.cumsum()[J] - arr.cumsum()[I]

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
    print(factor.dropna(how='all').corrwith((rets * (masks/masks)).dropna(how='all')).mean()/ factor.dropna(how='all').corrwith((rets * (masks/masks)).dropna(how='all')).std())
    
    factor_ratio = factor.rank(method = 'first',pct = True)
    factor_group = [factor_ratio[(factor_ratio>down_line) & (factor_ratio<=up_line)] for down_line , up_line in group_lst]
    factor_group = [i/i for i in factor_group]
    rate_group = []
    
    for j, i in enumerate(factor_group):
        rate_group.append((rets*i).mean())
        print(j+1, i.count().mean())
        
    
    df = pd.DataFrame(rate_group,index = [('group' + str(i+1)) for i in range(group_num)])
    
    return df.T.iloc[:-1,:]


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
    ret_y =  (ret.dropna().sum())/len(ret.dropna())*250
    sharpe = (ret.dropna().sum())/len(ret.dropna())*250 /  ((ret.dropna().std()* 250**0.5))
    rankIC = Index_rets.rank(ascending=False).corrwith(Fdf.rank(ascending=False)).mean()
    rankIR = Index_rets.rank(ascending=False).corrwith(Fdf.rank(ascending=False)).mean() / Index_rets.rank(ascending=False).corrwith(Fdf.rank(ascending=False)).std()
    win_pct = ret[ret>0].dropna().shape[0] / ret.dropna().shape[0]
    gain_loss_rate = ret[ret>0].mean() / ret[ret<0].mean()
    
    print(f'yoy: {ret_y}','\n', f'mdd: {mdd}','\n', f'sharpe: {sharpe}','\n', f'rankIC: {rankIC}','\n', f'rankIR: {rankIR}','\n', f'win%: {win_pct}','\n', f'gain_loss%: {gain_loss_rate}')
    
    return Index_panel_select, ret


def linear_projector(_gr, F, winsize):
   
    ts_group_ret = _gr.dropna().mean()
    ex_ret_gr = _gr.apply(lambda x: x-_gr.mean(axis=1))

    Stz_X0 = (F.dropna(axis=1) - F.dropna(axis=1).mean()) / (F.dropna(axis=1).std())
    Stz_ex_ret_gr0 = ((ex_ret_gr.T - ex_ret_gr.T.mean()) / ex_ret_gr.T.std())

    win = winsize
    box = []

    newF = []
    for i,d in tqdm(zip(range(len(Stz_X0.T)-win), Stz_X0.T.index[win:])):
        Stz_X = Stz_X0.iloc[:,0+i:win+i]
        Stz_ex_ret_gr = Stz_ex_ret_gr0.iloc[:,0+i:win+i] 
        g_ret = Stz_ex_ret_gr.mean(axis=1)
        Stz_ex_ret_gr = Stz_ex_ret_gr

        Ridx = Stz_ex_ret_gr.mean(axis=1).rank().sort_values().index
        dct = {
            int(Ridx[0][-1]):1,# 组别:收益最小组
            int(Ridx[1][-1]):2,
            int(Ridx[2][-1]):3,
            int(Ridx[3][-1]):4,
            # int(Ridx[4][-1]):5,
            # int(Ridx[5][-1]):6,
            # int(Ridx[6][-1]):7,
        }

        Xd = Stz_X0.loc[:,d]
        if np.isnan(Xd.rank(method='first').values[1]):
            continue
        
        vs = []
        for v in Xd.rank(method='first'):
            try:
                newX = dct[v] 
            except:
                display(dct)
            vs.append(newX)

        newXd = pd.Series(vs,Xd.index).rename(d)
        newF.append(newXd)
        
    display(newXd)

    return pd.concat(newF,axis=1)

def read_Factor(trd_dates, name:str, df=None)->pd.DataFrame:
    if df is None:
        # 因子shift 1
        F = pd.read_csv(r'factors\DailyFactor' + '/' + name + '.csv',index_col=0).shift(1,axis=1)
        F = _Standlize(F)
    else:
        F = df.shift(1,axis=1)
        F = _Standlize(F)
    Index_factor_panel = pd.concat([
        (weights50 * F).sum().rename('50'),
        (weights300 * F).sum().rename('300'),
        (weights500 * F).sum().rename('500'),
        (weights1000 * F).sum().rename('1000'),
        # (weights100sz * F).sum().rename('sz100'),
        # (weights50kc * F).sum().rename('kc50'),
        # (weights2000 * F).sum().rename('2000')
    ],axis=1).T.loc[:,trd_dates]
    
    
    Z_panel = _Standlize(Index_factor_panel)
    return Z_panel


def read_Factor1(trd_dates, name:str, df=None)->pd.DataFrame:
    if df is None:
        # 因子shift 1
        F = pd.read_csv(r'factors\DailyFactor' + '/' + name + '.csv',index_col=0).shift(1,axis=1)
        F = _Standlize(F)
    else:
        F = df.shift(1,axis=1)
        F = _Standlize(F)
    Index_factor_panel = pd.concat([
        (weights50/weights50 * F).mean().rename('50'),
        (weights300/weights300 * F).mean().rename('300'),
        (weights500/weights500 * F).mean().rename('500'),
        (weights1000/weights1000 * F).mean().rename('1000'),
        # (weights100sz * F).sum().rename('sz100'),
        # (weights50kc * F).sum().rename('kc50'),
        # (weights2000 * F).sum().rename('2000')
    ],axis=1).T.loc[:,trd_dates]
    
    
    Z_panel = _Standlize(Index_factor_panel)
    return Z_panel