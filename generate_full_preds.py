import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from data.datasets import ForecastDataset
from models import get_model
from utils.ops import to_tensor, default_device

'''
@torch.no_grad()
def generate_full_preds(model, dataset, batch_size=32, save_dir=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, trues = [], []

    for batch in loader:
        x, y, x_time, y_time = map(to_tensor, batch)
        forecast = model(x, x_time, y_time)
        if isinstance(forecast, tuple):  # 处理多输出模型
            forecast = forecast[1]
        preds.append(forecast.cpu().numpy())
        trues.append(y.cpu().numpy())

    # 合并结果
    preds = np.concatenate(preds, axis=0)  # 形状: [样本数, 5, 1]
    trues = np.concatenate(trues, axis=0)

    # --- 计算指标 ---
    # 1. 计算每个时间步的RMSE
    rmse_steps = np.sqrt(np.mean((preds - trues) ** 2, axis=(0, 2)))  # 形状: (5,)

    # 2. 计算每个时间步的MAE
    mae_steps = np.mean(np.abs(preds - trues), axis=(0, 2))  # 新增MAE计算

    # 3. 计算每个时间步的相关系数
    corr_steps = np.array([
        pearsonr(preds[:, step, 0], trues[:, step, 0])[0]
        for step in range(5)
    ])  # 形状: (5,)

    # --- 打印结果 ---
    print("\n===== 预测指标 =====")
    for step in range(5):
        print(
            f"步{step + 1} RMSE: {rmse_steps[step]:.4f} | MAE: {mae_steps[step]:.4f} | 相关系数: {corr_steps[step]:.4f}")

    return rmse_steps, mae_steps, corr_steps  # 现在返回三个指标

    return rmse_steps, corr_steps  # 返回两个(5,)数组
'''




import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.ops import to_tensor

@torch.no_grad()
def generate_full_preds(model, dataset, batch_size=32, save_dir='storage/full_pred'):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    preds, trues = [], []

    for batch in loader:
        x, y, x_time, y_time = map(to_tensor, batch)
        out = model(x, x_time, y_time)
        if isinstance(out, tuple):
            out = out[1]
        preds.append(out.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'full_preds.npy'), preds)
    np.save(os.path.join(save_dir, 'full_trues.npy'), trues)
    print(f"✅ 保存完成：{save_dir}")
