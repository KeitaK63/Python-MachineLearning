import matplotlib.pyplot as plt
import numpy as np

# 乱数種を生成
np.random.seed(0)
# 標準正規分布に従う200行2列の行列
X_xor = np.random.randn(200, 2)
# 2つの引数に対して排他的論理和を実行
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
# 値が真の場合は1を、偽なら-1を割り当てる
print(X_xor)
y_xor = np.where(y_xor, 1, -1)
# ラベル1を青、-1を赤でプロットする
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./figures/xor.png', dpi=300)
plt.show()
