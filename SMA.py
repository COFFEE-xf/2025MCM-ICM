#移动平均能够去除时间序列的短期波动，使得数据变得平滑，从而可以方便看出序列的趋势特征。
#尤其在金融领域，移动平均线作为一种计算简单、易于解释的趋势性指标，可以从中看出市场的趋势和倾向。

import pandas as pd
import akshare as ak

df = ak.stock_zh_a_hist(symbol="000858", start_date="20211008", end_date='20211018')
df = df.set_index('日期')
df.index = pd.to_datetime(df.index)
df = df[['收盘']]

df['SMA_3'] = df['收盘'].rolling(window=3).mean()
df['SMA_5'] = df['收盘'].rolling(window=5).mean()

print(df)