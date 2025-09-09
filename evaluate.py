import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import os
from typing import Dict, Tuple

# 设置中文显示和字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12


def load_data(preds_path: str, trues_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载预测和真实值数据"""
    preds = np.load(preds_path).squeeze(-1)  # shape: [N, horizon]
    trues = np.load(trues_path).squeeze(-1)
    assert preds.shape == trues.shape, "预测和真实值形状不一致!"
    return preds, trues


def calculate_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    mask = ~np.isnan(true) & ~np.isnan(pred)  # 处理NaN值
    true, pred = true[mask], pred[mask]

    rmse = np.sqrt(np.mean((true - pred) ** 2))
    mae = np.mean(np.abs(true - pred))
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)

    # 仅当数据无零值时计算MAPE
    if np.all(true != 0):
        mape = 100 * np.mean(np.abs((true - pred) / true))
    else:
        mape = np.nan

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }


def statistical_tests(errors: np.ndarray) -> Dict[str, float]:
    """执行统计检验"""
    # Friedman检验（多步长比较）
    friedman_stat, friedman_p = stats.friedmanchisquare(*[errors[:, i] for i in range(errors.shape[1])])

    # 第1天 vs 第5天的Wilcoxon配对检验
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(errors[:, 0], errors[:, -1])

    return {
        'Friedman_p': friedman_p,
        'Wilcoxon_p_Day1_vs_Day5': wilcoxon_p
    }


def plot_error_trend_with_ci(metrics: Dict[int, Dict[str, float]],
                             errors: np.ndarray,
                             test_results: Dict[str, float],
                             save_dir: str = None):
    """
    绘制带置信区间的误差趋势图
    :param metrics: 各步长的指标字典
    :param errors: 误差矩阵 [N, horizon]
    :param test_results: 统计检验结果
    :param save_dir: 保存路径
    """
    horizons = sorted(metrics.keys())
    rmse_values = [metrics[h]['RMSE'] for h in horizons]

    # 计算RMSE的95%置信区间（Bootstrap法）
    n_boot = 1000
    boot_rmse = np.zeros((n_boot, len(horizons)))

    for i in range(n_boot):
        sample_idx = np.random.choice(len(errors), size=len(errors), replace=True)
        boot_rmse[i] = [np.sqrt(np.mean(errors[sample_idx, h] ** 2)) for h in range(len(horizons))]

    ci_low, ci_high = np.percentile(boot_rmse, [2.5, 97.5], axis=0)

    # 绘图设置
    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # 主曲线和置信区间
    ax.plot(horizons, rmse_values, marker='o', linestyle='-',
            color='#1f77b4', linewidth=2, markersize=8, label='RMSE')
    ax.fill_between(horizons, ci_low, ci_high, color='#1f77b4', alpha=0.2, label='95% 置信区间')

    # 标注统计检验结果
    test_text = (f"Friedman检验 p={test_results['Friedman_p']:.3f}\n"
                 f"Day1 vs Day5 Wilcoxon p={test_results['Wilcoxon_p_Day1_vs_Day5']:.3f}")
    ax.text(0.02, 0.95, test_text, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            verticalalignment='top')

    # 坐标轴和标题
    ax.set_xlabel("预测步长 (天)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("预测误差随步长的变化趋势", fontsize=14, pad=20)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)

    # 调整布局
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'error_trend_with_ci.png'),
                    dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_violin(errors: np.ndarray, save_dir: str = None):
    """绘制误差分布小提琴图"""
    df = pd.DataFrame(errors, columns=[f'Day{i + 1}' for i in range(errors.shape[1])])

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # 小提琴图
    sns.violinplot(data=df, inner="quartile", palette="Blues", ax=ax)

    # 标注中位数
    for i, col in enumerate(df.columns):
        median_val = df[col].median()
        ax.text(i, median_val, f"{median_val:.2f}",
                ha='center', va='bottom', color='red', fontweight='bold')

    # 坐标轴和标题
    ax.set_xlabel("预测步长", fontsize=12)
    ax.set_ylabel("预测误差", fontsize=12)
    ax.set_title("各步长误差分布（小提琴图）", fontsize=14, pad=20)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'error_violin.png'),
                    dpi=300, bbox_inches='tight')
    plt.close()


def plot_pred_vs_true_scatter(preds: np.ndarray,
                              trues: np.ndarray,
                              metrics: Dict[int, Dict[str, float]],
                              save_dir: str = None):
    """绘制预测-真实值散点图（分步长）"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    fig.suptitle("预测值与真实值散点对比（按预测步长）", y=1.05, fontsize=14)

    for h in range(5):
        ax = axes[h]
        # 散点图
        ax.scatter(trues[:, h], preds[:, h], alpha=0.5, s=10, color='#1f77b4')

        # 对角线参考线
        ax.plot([trues.min(), trues.max()], [trues.min(), trues.max()],
                'r--', linewidth=1)

        # 标题和标签
        ax.set_title(f"Day {h + 1}\n(R²={metrics[h + 1]['R²']:.2f}, RMSE={metrics[h + 1]['RMSE']:.2f}")
        ax.set_xlabel("真实值")
        if h == 0:
            ax.set_ylabel("预测值")
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_dir:
        plt.show()
    plt.close()


def save_metrics_table(metrics: Dict[int, Dict[str, float]], save_path: str):
    """保存指标为CSV表格"""
    df = pd.DataFrame.from_dict(metrics, orient='index')
    df.index.name = 'Horizon (day)'
    df.to_csv(save_path, float_format='%.4f')
    print(f"指标已保存至: {save_path}")


def main():
    # 配置路径
    preds_path = 'storage/full_pred/full_preds.npy'
    trues_path = 'storage/full_pred/full_trues.npy'
    save_dir = 'results/metrics'
    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载数据
    preds, trues = load_data(preds_path, trues_path)
    horizons = preds.shape[1]

    # 2. 计算各步长指标
    metrics = {}
    errors = np.zeros_like(preds)  # 存储所有误差值

    for h in range(horizons):
        errors[:, h] = trues[:, h] - preds[:, h]
        metrics[h + 1] = calculate_metrics(trues[:, h], preds[:, h])

    # 3. 统计检验
    test_results = statistical_tests(errors)
    print("\n统计检验结果:")
    print(f"Friedman检验p值: {test_results['Friedman_p']:.4f}")
    print(f"第1天vs第5天Wilcoxon检验p值: {test_results['Wilcoxon_p_Day1_vs_Day5']:.4f}")

    # 4. 可视化
    plot_error_trend_with_ci(metrics, errors, test_results, save_dir)
    plot_error_violin(errors, save_dir)
    plot_pred_vs_true_scatter(preds, trues, metrics, save_dir)

    # 5. 保存结果
    save_metrics_table(metrics, os.path.join(save_dir, 'metrics_table.csv'))

    # 6. 打印关键结果
    print("\n各步长指标:")
    print(pd.DataFrame.from_dict(metrics, orient='index').round(4))


if __name__ == '__main__':
    main()