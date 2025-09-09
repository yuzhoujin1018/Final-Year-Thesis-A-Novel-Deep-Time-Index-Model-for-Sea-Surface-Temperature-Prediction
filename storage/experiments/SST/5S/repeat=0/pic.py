import numpy as np
import matplotlib.pyplot as plt

trues = np.load('trues.npy')
preds = np.load('preds.npy')

# 假设是 shape: [样本数, 时间步]
sample_idx = 0
x = np.arange(preds.shape[1])

plt.figure(figsize=(10, 5))
plt.plot(x, trues[sample_idx], label='true', color='blue')
plt.plot(x, preds[sample_idx], label='pred', color='orange')
plt.xlabel("天数")
plt.ylabel("SST/°C")
plt.title("STL组合预测结果")
plt.legend()
plt.grid(True)
plt.show()
