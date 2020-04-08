import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ClassDecisionRegions import plot_decision_regions
from ClassLogisticRegression import LogisticRegression
#SSl認証が正しくなく下記2行がないとPythonのシステム上エラーとなる？らしい（未解決）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#データのダウンロード
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

# 1－100行目の目的関数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを1，Iris-virginicaを0に変換
y = np.where(y == 'Iris-setosa', 1, 0)

# 1-100行目の1，3列目の抽出
X = df.iloc[0:100, [0, 2]].values
#####################################
# 以下標準偏差に関する変更
# standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = LogisticRegression(n_iter=500, eta=0.2)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Logistic Regression - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./adaline_2.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
# plt.savefig('./adaline_3.png', dpi=300)
plt.show()
