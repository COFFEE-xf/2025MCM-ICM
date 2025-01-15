# 对于金融时间序列，波动率往往具有以下特征：
#
# （1）存在波动率聚集现象。 即波动率在一段时间上高，一段时间上低。
# （2）波动率以连续时间变化，很少发生跳跃
# （3）波动率不会发散到无穷，波动率往往是平稳的
# （4）波动率对价格大幅上升和大幅下降的反应是不同的，这个现象为杠杆效应
#
# 条件波动性（Conditional Volatility） 是指在时间序列数据中，
# 一个时刻的波动性（方差）依赖于该时刻之前的数据。
# 换句话说，条件波动性表示了时间序列数据的波动性在不同时刻是不同的，且这种变化是基于之前的观测值而来的。
# ARCH（Autoregressive Conditional Heteroskedasticity）和
# GARCH（Generalized Autoregressive Conditional Heteroskedasticity）模型
# 就是用来建模时间序列数据中的条件波动性的方法。
# 这些模型认为时间序列数据的方差是随着时间变化的，并且可以通过过去的观测值来预测。

# ARCH的基本原理
# 在传统计量经济学模型中，干扰项的方差被假设为常数。
# 但是许多经济时间序列呈现出波动的集聚性，在这种情况下假设方差为常数是不恰当的。
# ARCH模型将当前一切可利用信息作为条件，并采用某种自回归形式来刻划方差的变异，
# 对于一个时间序列而言，在不同时刻可利用的信息不同，而相应的条件方差也不同，
# 利用ARCH模型，可以刻划出随时间而变异的条件方差。

# 虽然ARCH模型简单，但为了充分刻画收益率的波动率过程，往往需要很多参数，
# 例如上面用到ARCH(10)模型，有时会有更高的ARCH(m)模型。
# 因此，Bollerslev(1986)年提出了一个推广形式，称为广义的ARCH模型（GARCH）
# GARCH（Generalized Autoregressive Conditional Heteroskedasticity）模型
# 是一种用于建模金融时间序列数据中异方差性的统计模型。它是
# ARCH（Autoregressive Conditional Heteroskedasticity）模型的扩展，
# 通过引入过去的方差（方差残差）的高阶项来更准确地描述时间序列数据中的方差结构。



