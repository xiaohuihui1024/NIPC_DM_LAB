# -*- coding: utf-8 -*-
'''
学习器文件,生成模拟数据，调用 knn.py 类 做分类
完成精度评价
'''
import numpy as np
from my_classifier.knn import KNN
import matplotlib as plt
# data generation
# 为了保证每次数据模拟结果一样，把随机种写死
np.random.seed(272)
# 模拟数据：0,1分类简单问题，两个类别
# 为了便于展示，特征空间设置为两个维度x1,x2

# 第1组数据，生成正态分布随机数
data_size_1 = 300
x1_1 = np.random.normal(loc=5.0, scale=1.0, size=data_size_1)
x2_1 = np.random.normal(loc=4.0, scale=1.0, size=data_size_1)
y_1 = [0]*data_size_1

# 第2组数据，生成正态分布随机数
data_size_2 = 400
x1_2 = np.random.normal(loc=10.0, scale=2.0, size=data_size_2)
x2_2 = np.random.normal(loc=8.0, scale=2.0, size=data_size_2)
y_2 = [1]*data_size_2

# 把两组数据放在一起，构成完整数据
x1 = np.concatenate((x1_1, x1_2), axis=0) # 0号坐标轴
x2 = np.concatenate((x2_1, x2_2), axis=0)

# 默认生成的是行向量，转化为列向量后再合并，x合并后是2列
x = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))
y = np.concatenate((y_1, y_2), axis=0)

# 数据已经生成，但数据前300个是类别1，后400个是类别2，不太自然，需要洗牌，洗牌其实就是把索引打乱
data_size_all = data_size_1 + data_size_2
# 生成随机索引序列
shuffled_index = np.random.permutation(data_size_all)
# 进行洗牌
x = x[shuffled_index]
y = y[shuffled_index]

# 切分训练集和测试集，由于已经洗牌，找切分点即可
split_index = int(data_size_all * 0.7)
x_train = x[:split_index]
y_train = y[:split_index]
x_test = x[split_index:]
y_test = y[split_index:]

# 归一化到0-1
x_train = (x_train - np.min(x_train, axis=0)) / \
          (np.max(x_train, axis=0) - np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0)) / \
          (np.max(x_test, axis=0) - np.min(x_test, axis=0))

# knn classifier
clf = KNN(k=3)
clf.fit(x_train, y_train)
score_train = clf.score()
print('train accuracy: {:.3}'.format(score_train))

y_test_pred = clf.predict(x_test)
print('test accuracy: {:.3}'.format(clf.score(y_test, y_test_pred)))
# train accuracy: 0.994
# test accuracy: 0.976

exit()

# visual data
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='.')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='.')


# x是二维向量，纵向全要
plt.scatter(x[:, 0], x[:, 1], c=y)

# 绘图
# 效果：两组数据有明显分离，但中间还要混杂地带，学习器有些小难度
plt.scatter(x1_1, x2_1)
plt.scatter(x1_2, x2_2)
plt.show()
