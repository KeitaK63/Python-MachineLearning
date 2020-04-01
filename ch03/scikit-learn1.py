# Irisデータセットをロード
# 3,4列目の特徴量を抽出
# クラスラベルを習得
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print("Class labels:", np.unique(y))

# トレーニングセットとテストデータに分割
# 全体の30%をテストデータにする
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# トレーニングデータの平均と標準偏差を計算
# 平均と標準偏差を用いて標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# エボック数40,学習率0.1でパーセプトロンのインスタンスを生成
# トレーニングデータをモデルに適合させる
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.01, random_state=0, shuffle=True, n_iter_no_change=40)#n_iter=40,
ppn.fit(X_train_std, y_train)

# テストデータで予測を実施
# 誤分類のサンプルの個数を表示
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

# 分類の正解率を表示
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# プロット
from ClassDecisionRegions import plot_decision_regions
import matplotlib.pyplot as plt
# トレーニングデータとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))
# トレーニングデータとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))
# 決定領域のプロット
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))
# 軸ラベルの設定
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
# 凡例の設定
plt.legend(loc='upper left')
# グラフの表示
plt.show()
