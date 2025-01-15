# ARIMA模型是时间序列分析中广泛应用到的一个模型，分为以下三个部分：
# 1.AR(自回归)部分：即使用时间序列自身的过去值来预测当前值
# 2.I(积分)部分：通过对数据进行差分处理，使得非平稳时间序列变为平稳序列
# 3.MA(移动平均)部分：使用过去的预测误差来修正预测

# 首先来看AR模型：
import pandas as pd
import numpy as np
from statsmodels.tsa.api import AutoReg

# 创建数据框
data = pd.DataFrame({
    'temperature': [20, 22, 21, 23, 22, 21, 20]
})

# 分割数据，留出最后一天用于预测
train_data = data['temperature'][:-1]
test_data = data['temperature'][-1:]

# 创建并拟合 AR(1) 模型
model = AutoReg(train_data, lags=1, trend='c')
results = model.fit()

# 预测第7天的气温
forecast = results.predict(start=len(train_data), end=len(train_data), dynamic=False)
print("第7天气温预测为:", forecast[6])

# ---- 输出 -----
# 第7天气温预测为: 21.9615
# ----------------------------------------------------------------------------------------------------

# 接下来来看MA模型：

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 设置字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 生成示例数据
np.random.seed(42)
data = np.random.randn(100)

# 拟合 MA 模型
model = ARIMA(data, order=(0, 0, 1))  # 这里 (p, d, q) = (0, 0, 1) 表示 MA(1) 模型
model_fit = model.fit()

# 预测下一个值
forecast = model_fit.forecast(steps=1)
print(f"预测的下一个值: {forecast[0]}")

# 可视化原始数据和预测值
plt.plot(data, label='原始数据')
plt.axvline(x=len(data), color='gray', linestyle='--')
plt.plot(len(data), forecast, 'ro', label='预测值')
plt.legend()
plt.title('MA 模型预测')
plt.show()

# -----------输出------------
# 预测的下一个值: -0.10097
# ----------------------------------------------------------------------------------------------------

# 接下来就是ARIMA模型：
# ARIMA 模型综合 AR 模型和 MA 模型，基本思想是：
# 一个时间点上的数据值既受过去一段时间内的数据值影响，也受过去一段时间内的偶然事件的影响，
# 这就是说，ARIMA模型假设：数据值是围绕着时间的大趋势而波动的，其中趋势是受历史标签影响构成的，
# 波动是受一段时间内的偶然事件影响构成的，且大趋势本身不一定是稳定的。

# 那么ARIMA模型中的I部分又怎么理解呢？

# I 表示差分，用于使非平稳时间序列达到平稳，通过一阶或者二阶等差分处理，
# 差分的基本思想是计算时间序列相邻观测值之间的差。
# 通过差分，可以消除时间序列中的趋势和季节性，从而实现平稳化。

# 差分的阶数：
# 前面，我们介绍了一阶差分。然而，实际上，差分的阶数可以是任何正整数。
# 差分的阶数就是我们需要进行多少次差分操作才能得到一个平稳序列。
# 具体地说就是：二阶差分就是对一阶差分后的序列再次进行差分，三阶差分就是对二阶差分后的序列再次进行差分。
# 需要注意的是：差分的阶数不会很大，在平时工作中使用 ARIMA 模型时，差分的阶数一般为 0、1、2，几乎不会是更大的了。

# 在 ARIMA(p, d, q) 模型中存在三个参数 p、d、q ，根据前面我们所介绍的，它们分别具有以下作用：
# p：表示 AR 自回归模型的阶数，即 AR 自回归模型考虑的过去多久的数据的线性组合；
# d：表示差分的阶数，即这个数据使用几阶差分进行处理成平稳数据；
# q：表示 MA 移动平均模型的阶数，即 MA 移动平均模型考虑过去多久的白噪声影响；

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 示例数据：月度销售数据
data = {
    'Month': ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05',
              '2020-06', '2020-07', '2020-08', '2020-09', '2020-10',
              '2020-11', '2020-12'],
    'Sales': [200, 220, 250, 230, 300, 280, 310, 320, 330, 360, 400, 450]
}

# 创建 DataFrame
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# 构建和拟合 ARIMA 模型 (p=12, d=1, q=12)
model = ARIMA(df['Sales'], order=(12, 1, 12))
model_fit = model.fit()

# 进行预测
forecast = model_fit.forecast(steps=6)  # 预测未来 6 个月
print('Forecasted values:', forecast)

# 绘制预测结果
plt.figure(figsize=(10, 4))
plt.plot(df['Sales'], label='原始销量', marker='o')
plt.plot(pd.date_range(start='2020-12-31', periods=6, freq='M'), forecast, label='预测', color='red', marker='o')
plt.title('ARIMA 预测')
plt.xlabel('月份')
plt.ylabel('销量')
plt.legend()
plt.grid()
plt.show()

