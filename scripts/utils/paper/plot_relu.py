import numpy as np
import matplotlib.pyplot as plt

# データの準備
x = np.linspace(-10, 10, 1000)
y = np.maximum(0, x)  # ReLU関数

# プロット
plt.figure(figsize=(16, 9))
plt.plot(x, y, linewidth=15, color='#1f77b4', label='ReLU(x)')
plt.grid(True, color='gray', alpha=0.5, linewidth=3)

# x=0に縦線を追加
plt.axvline(x=0, color='black', linewidth=10, linestyle='-')

# 軸の設定
ax = plt.gca()
ax.tick_params(width=2, length=6, labelsize=12, colors='black')

plt.xlim(-11, 11)
plt.ylim(-0.5, 10.5)

# 背景を白に
ax.set_facecolor('white')
plt.gcf().patch.set_facecolor('white')

# 保存
plt.savefig('relu_function.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('relu_function.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print('ReLU関数の画像を保存しました: relu_function.png, relu_function.pdf')
plt.close()
