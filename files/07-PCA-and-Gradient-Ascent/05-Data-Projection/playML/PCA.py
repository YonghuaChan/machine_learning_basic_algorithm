import numpy as np


class PCA:

    def __init__(self, n_components):
        """初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None  # 通过用户传来的数据计算得出的, 不是用户传进来的

    def fit(self, X, eta=0.01, n_iters=1e4):
        # 传入X
        # 因为使用梯度上升法, 所以传入eta
        # n_iter最大迭代次数默认设为10000
        """获得数据集X的前n个主成分"""
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"

        def demean(X):  # 均值归零化
            return X - np.mean(X, axis=0)

        def f(w, X):  # 求目标函数f
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):  # 求df
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):  # 求一个w的方向
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):  # 梯度上升法

            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break

                cur_iter += 1

            return w

        X_pca = demean(X)  # 首先要进行均值归零化操作
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))  # 初始化空矩阵, 行数=n_components, 列数就是样本X的列数
        for i in range(self.n_components):  # 设置一个循环
            initial_w = np.random.random(X_pca.shape[1])  # 每一次设置一个初始的搜索方向
            w = first_component(X_pca, initial_w, eta, n_iters)  # 然后使用梯度上升法搜索此时的X_pca对应的主成分
            self.components_[i,:] = w  # 然后将这个w放在components_的第i行的位置

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w  # 然后在X_pca中减去我们的样本在刚刚求的w方向上所有的分量, 形成一个新的x_pca, 进行下一次循环, 以此类推

        return self

    def transform(self, X):
        """将给定的X，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射回原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]

        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components

