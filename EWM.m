% 熵权法是利用信息熵这一工具，计算出各个指标的权重，为多指标综合评价提供依据
% 根据信息熵的定义，对于某项指标，可以利用熵值来判断某个指标的离散程度，其信息熵越小，指标的离散程度越大，该指标对综合评价的影响就越大
% 熵权法是一种客观分析法，相对于层次分析法等主观赋值法的精度更高，客观性更强

% 示例来自CSDN博客：https://blog.csdn.net/qq_48774513/article/details/120636330
 
X=[124.3000    2.4200   25.9800   19.0000    3.1000   79.0000   54.1000    6.1400    3.5700   64.0000
  134.7000    2.5000   21.0000   19.2500    3.3400   84.0000   53.7000    6.7400    3.5500   64.9600
  193.3000    2.5600   29.2600   19.3100    3.4500   92.0000   54.0000    7.1800    3.4000   65.6500
  118.6000    2.5200   31.1900   19.3600    3.5700  105.0000   53.9000    8.2000    3.2700   67.0100
   94.9000    2.6000   28.5600   19.4500    3.3900  108.0000   53.6000    8.3400    3.2000   65.3400
  123.8000    2.6500   28.1200   19.6200    3.5800  108.0000   53.3000    8.5100    3.1000   66.9900];

% 熵权法的主要步骤：
% 1.数据标准化：将各个指标进行去量纲化处理（所给数据已完成标准化，此步略）
% 2.计算第j个指标第i个项目的数值比重Pij：
    clear
    clc
    
    [n, m] = size(X);
    for i = 1 : n
        for j = 1 : m
            p(i, j) = X(i, j) / sum(X(:, j));
        end
    end
% 3.求出第j个指标的熵值e(j):
    k = 1 / log(n);
    for j = 1 : m
        e(j) = -k * sum(p(:, j).*log(p(:, j)));
    end
% 4.确定各指标的权重(通过计算信息冗余度)
    d = ones(1, m) - e;
    w = d./sum(d)

    

