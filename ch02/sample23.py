import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ClassPerceptron import Perceptron
from ClassDecisionRegions import plot_decision_regions
from ClassAdalineSGD import AdalineSGD

#データのダウンロード
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

# 1－100行目の目的関数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1，Iris-virginicaを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)

# 1-100行目の1，3列目の抽出
X = df.iloc[0:100, [0, 2]].values

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('./adaline_4.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
# plt.savefig('./adaline_5.png', dpi=300)
plt.show()
