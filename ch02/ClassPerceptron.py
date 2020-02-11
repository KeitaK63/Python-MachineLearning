import numpy as np


class Perceptron(object):
    """Perceptron classifier.(パーセプトロンの分類機)

    Parameters(パラメータ)
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)，(学習率)
    n_iter : int
        Passes over the training dataset.，(トレーニングデータのトレーニング回数)

    Attributes(属性)
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.(トレーニングデータに適合させる)

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # 重みw1....wmの更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # 重みw0の更新
                self.w_[0] += update
                # 重みの更新が0でない場合は誤差分類としてカウント
                errors += int(update != 0.0)
            # 反復回数ごとｎ誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input(総入力を計算)"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step(1ステップ後のクラスラベルを返す)"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
