import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib as mpl

# 设置字体为宋体
mpl.rcParams['font.sans-serif'] = ['SimSun']
mpl.rcParams['axes.unicode_minus'] = False

# 加载数据
trues = np.load('full_trues.npy').squeeze(-1)  # (3151, 10)
preds = np.load('full_preds.npy').squeeze(-1)  # (3151, 10)

# 展平成一维
true_flat = trues.flatten()
pred_flat = preds.flatten()

# 计算 R² 分数
r2 = r2_score(true_flat, pred_flat)

# 绘图
plt.figure(figsize=(6, 6))
plt.scatter(true_flat, pred_flat, s=5, alpha=0.5, label='数据点')
plt.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], 'r--', label='理想拟合线')

plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('真实值 vs 预测值 散点图')

# 左上角显示 R²
plt.text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='left')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
