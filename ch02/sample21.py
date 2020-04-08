import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ClassPerceptron import Perceptron
from ClassDecisionRegions import plot_decision_regions
#SSl認証が正しくなく下記2行がないとPythonのシステム上エラーとなる？らしい（未解決）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#データのダウンロード
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

# 1－100行目の目的関数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1，Iris-virginicaを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)

# 1-100行目の1，3列目の抽出
X = df.iloc[0:100, [0, 2]].values


# データのプロット
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')
# 軸ラベルの
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('./images/02_06.png', dpi=300)
plt.show()


# エボックに関するプロット
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.tight_layout()
# plt.savefig('./perceptron_1.png', dpi=300)
plt.show()


# 決定境界のプロット
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('./perceptron_2.png', dpi=300)
plt.show()
