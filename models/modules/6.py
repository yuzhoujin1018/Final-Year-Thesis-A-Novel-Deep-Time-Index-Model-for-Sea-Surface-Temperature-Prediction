import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.patches import Rectangle

# 设置中文和字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 10})

# 创建数据
np.random.seed(42)
n_points = 250
x = np.linspace(0, 10, n_points)
y = (np.sin(x * 2) +
     0.5 * np.sin(x * 0.5) +
     0.1 * np.random.normal(size=n_points))

# 平滑曲线
spline = interpolate.make_interp_spline(x, y, k=3)
x_smooth = np.linspace(x.min(), x.max(), 1000)
y_smooth = spline(x_smooth)

# 计算分割点
split_idx = int(len(x_smooth) * 0.7)
split_x = x_smooth[split_idx]

# 创建图形（垂直排列两个子图）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# ===== 第一个子图：历史值模型（加上测试集输入输出示意） =====
ax1.plot(x_smooth, y_smooth, color='gray', alpha=0.3, linewidth=1)
ax1.plot(x_smooth[:split_idx], y_smooth[:split_idx],
         color='royalblue', linewidth=2, label='训练集')
ax1.plot(x_smooth[split_idx:], y_smooth[split_idx:],
         color='crimson', linewidth=2, label='测试集')

# ---- 训练集的输入/输出窗口 ----
window_size = 100
pred_size = 25
start_pos = 150  # 训练集窗口起点

# 输入窗口
input_window = x_smooth[start_pos:start_pos+window_size]
ax1.plot(input_window, y_smooth[start_pos:start_pos+window_size],
         color='darkblue', linewidth=3, alpha=0.7)
# 输出窗口
output_window = x_smooth[start_pos+window_size:start_pos+window_size+pred_size]
ax1.plot(output_window, y_smooth[start_pos+window_size:start_pos+window_size+pred_size],
         color='darkred', linestyle='--', linewidth=3, alpha=0.7)

# 添加虚线框
ax1.axvline(x=input_window[0], color='darkblue', linestyle='--', linewidth=1)
ax1.axvline(x=input_window[-1], color='darkblue', linestyle='--', linewidth=1)
ax1.axvline(x=output_window[0], color='darkred', linestyle='--', linewidth=1)
ax1.axvline(x=output_window[-1], color='darkred', linestyle='--', linewidth=1)

# 标注
ax1.text((input_window[0] + input_window[-1]) / 2, min(y_smooth)-0.4, '输入', ha='center', fontsize=10, color='darkblue')
ax1.text((output_window[0] + output_window[-1]) / 2, min(y_smooth)-0.4, '输出', ha='center', fontsize=10, color='darkred')

# ---- 测试集的输入/输出窗口 ----
test_start_pos = split_idx + 100  # 测试集窗口起点

# 测试输入窗口
test_input_window = x_smooth[test_start_pos:test_start_pos+window_size]
ax1.plot(test_input_window, y_smooth[test_start_pos:test_start_pos+window_size],
         color='darkblue', linewidth=3, alpha=0.7)

# 测试输出窗口
test_output_window = x_smooth[test_start_pos+window_size:test_start_pos+window_size+pred_size]
ax1.plot(test_output_window, y_smooth[test_start_pos+window_size:test_start_pos+window_size+pred_size],
         color='darkred', linestyle='--', linewidth=3, alpha=0.7)

# 添加虚线框
ax1.axvline(x=test_input_window[0], color='darkblue', linestyle='--', linewidth=1)
ax1.axvline(x=test_input_window[-1], color='darkblue', linestyle='--', linewidth=1)
ax1.axvline(x=test_output_window[0], color='darkred', linestyle='--', linewidth=1)
ax1.axvline(x=test_output_window[-1], color='darkred', linestyle='--', linewidth=1)

# 标注
ax1.text((test_input_window[0] + test_input_window[-1]) / 2, min(y_smooth)-0.4, '输入', ha='center', fontsize=10, color='darkblue')
ax1.text((test_output_window[0] + test_output_window[-1]) / 2, min(y_smooth)-0.4, '输出', ha='center', fontsize=10, color='darkred')



# 其他细节
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel('时间', fontsize=12)
ax1.set_ylabel('观测值', fontsize=12)
ax1.set_title('历史值模型：基于过去观测值预测未来值', fontsize=12, pad=10)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_ylim(min(y_smooth)-0.5, max(y_smooth)+0.5)
ax1.grid(True, linestyle=':', alpha=0.7)

# ===== 第二个子图：时间索引模型（根据你的要求修改） =====
ax2.plot(x_smooth, y_smooth, color='gray', alpha=0.3, linewidth=1.5)

# 选择训练集和测试集输入点
train_input_idx = [100, 200, 300]  # 任意选的训练输入点索引
test_input_idx = [750, 850, 950]   # 任意选的测试输入点索引

# 绘制输入点（蓝色训练，红色测试）
ax2.scatter(x_smooth[train_input_idx], [min(y_smooth)-0.3]*3,
            color='royalblue', s=80, marker='v', label='训练输入时间索引')
ax2.scatter(x_smooth[test_input_idx], [min(y_smooth)-0.3]*3,
            color='crimson', s=80, marker='v', label='测试输入时间索引')

# 绘制对应输出观测点
ax2.scatter(x_smooth[train_input_idx], y_smooth[train_input_idx],
            color='royalblue', edgecolors='black', s=80, label='训练输出观测值')
ax2.scatter(x_smooth[test_input_idx], y_smooth[test_input_idx],
            color='crimson', edgecolors='black', s=80, label='测试输出观测值')

# 连线（从输入时间索引到输出观测值）
for idx in train_input_idx:
    ax2.plot([x_smooth[idx], x_smooth[idx]], [min(y_smooth)-0.3, y_smooth[idx]], color='royalblue', linestyle='--')
for idx in test_input_idx:
    ax2.plot([x_smooth[idx], x_smooth[idx]], [min(y_smooth)-0.3, y_smooth[idx]], color='crimson', linestyle='--')



# 添加分割线
ax2.axvline(x=split_x, color='black', linestyle='--', linewidth=1.2)

ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('时间索引', fontsize=12)
ax2.set_ylabel('观测值', fontsize=12)
ax2.set_title('时间索引模型：建立时间索引到观测值的映射', fontsize=12, pad=10)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(min(y_smooth)-0.5, max(y_smooth)+0.5)
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()


