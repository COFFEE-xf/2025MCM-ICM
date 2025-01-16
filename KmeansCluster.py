# K值聚类是一种无监督的学习，事先不知道类别，自动将相似的对象归到同一个簇中

from sklearn.datasets import make_blobs

#参数：
# n_samples=100  样本数量
# n_features=2   特征数量
# centers=3      中心点

#返回值：
# X_train:  测试集
# y_train： 特征值

X_train,y_train = make_blobs(n_samples=100, n_features=2, centers=3)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train)

#参数
# n_clusters  将预测结果分为几簇

kmeans = KMeans(n_clusters=3)  # 获取模型
kmeans.fit(X_train)  #这里不需要给他答案 只把要分类的数据给他 即可

y_ = kmeans.predict(X_train)

plt.scatter(X_train[:,0],X_train[:,1],c=y_train) # 预测结果
plt.scatter(X_train[:,0],X_train[:,1],c=y_)  #原结果
