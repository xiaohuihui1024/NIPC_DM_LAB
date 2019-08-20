# 实现knn学习算法内部分类的函数
import numpy as np
import operator

class KNN(object):
    def __init__(self, k = 3):
        self.k = k

    # fit把x, y传进去,学习出一个模型
    # 而KNN并没有学习的过程，就是根据最近距离k个训练样本，得到推测值
    def fit(self, x, y):
        self.x = x
        self.y = y

    # 定义一个内部函数，计算任意两点间的距离的平方
    # 没有考虑每个特征的量纲，如果v1[0-100],v2[0-1]，数量级比较大，空间距离有问题
    # 需要把特征归一化
    def _square_distance(self, v1, v2):
        # 每个维度上相减后平方，再求和
        return np.sum(np.square(v1-v2))

    # 内部函数：投票机制
    def _vote(self, ys):
        # 有哪几个类可投,取ys中的独立的值
        ys_unique = np.unique(ys)
        vote_dict = {}
        for y in ys:
            if y not in vote_dict.keys():
                vote_dict[y] = 1
            else:
                vote_dict[y] += 1
        sorted_vote_dict = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_vote_dict[0][0]


    def predict(self, x):
        '''
        :param x: 多行的点，每一行的二维向量
        :return:
        '''
        y_pred = []
        # 对x的每一行进行循环
        # 计算x[i]与它最近的k个训练集中的点是哪几个
        for i in range(len(x)):
            # 当前要推测的x[i]和所有训练点的平方距离dist_arr
            dist_arr = [self._square_distance(x[i], self.x[j]) for j in range(len(self.x))]
            # 对dist_arr进行小到大排序，返回排序的索引
            sorted_index = np.argsort(dist_arr)
            # 提取top k
            top_k_index = sorted_index[:self.k]
            # 添加x[i]的预测值：k个点中投票机制
            y_pred.append(self._vote(ys=self.y[top_k_index]))
        return np.array(y_pred)

    # 精度
    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_pred = self.predict(self.x)
            y_true = self.y
        score = 0.0
        for i in range(len(y_true)):
            if (y_true[i] == y_pred[i]).all():
                score+=1
        score /= len(y_true)
        return score
