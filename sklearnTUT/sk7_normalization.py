# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# eg1
a = np.array([[10, 2.7, 3.6],
                     [-100, 5, -2],
                     [120, 20, 40]], dtype=np.float64)
print(a)
# 归一化 默认的方式
print(preprocessing.scale(a))
# 归一化 minmax_scale方式
# print(preprocessing.minmax_scale(a, feature_range=(-5,5)))


# eg2
X, y = make_classification(n_samples=300, n_features=2 , n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)

# X 300个样本，每个样本2维，y为类别标签，这里是0或者1
# 分析生成的数据
plt.scatter(X[:, 0], X[:, 1], c=y)
print(type(X))
print(X.dtype)
print(X.shape)
print(X[:, 0])
print(X[:, 0])
print(y)


plt.show()
X = preprocessing.scale(X)    # normalization step
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
clf = SVC()
clf.fit(X_train, y_train)
print()
print(clf.score(X_test, y_test))















