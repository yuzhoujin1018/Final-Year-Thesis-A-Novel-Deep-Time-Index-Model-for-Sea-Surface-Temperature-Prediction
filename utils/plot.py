import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif']=['SimSun']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams.update({'font.size': 10}) # 设置字体大小


def plot_horizon_steps(preds_path, trues_path, save_dir=None):
    preds = np.load(preds_path).squeeze(-1)  # [N, 5]
    trues = np.load(trues_path).squeeze(-1)
    horizon = preds.shape[1]

    # 计算每个预测步的实际数据点数量
    points_per_step = len(preds) // horizon

    for h in range(horizon):
        pred_series = preds[:, h]
        true_series = trues[:, h]

        # 只取对应时间步的数据点
        start_idx = h * points_per_step
        end_idx = (h + 1) * points_per_step
        pred_series = pred_series[start_idx:end_idx]
        true_series = true_series[start_idx:end_idx]

        x = np.arange(len(pred_series))

        plt.figure(figsize=(10, 5))
        plt.plot(x, true_series, label='真实值', color='blue', alpha=0.7, linewidth=1.5)
        plt.plot(x, pred_series, label='预测值', color='orange', alpha=0.7, linewidth=1.5)

        # 添加误差阴影
        plt.fill_between(
            x,
            true_series,
            pred_series,
            color='gray',
            alpha=0.5,
            label='误差'
        )
        plt.xlabel("天数")
        plt.ylabel("SST/℃")
        plt.legend()


        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'step_{h + 1}_forecast.png'))
        else:
            plt.show()