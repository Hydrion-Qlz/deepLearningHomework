import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('/data/zcd/hj/houseNumberRecoginzation/final/yolov5/runs/train/yolov5m/results.csv')
cleaned_keys = [key.strip() for key in data.keys()]
data.columns = cleaned_keys

# 设置绘图样式
plt.style.use('seaborn-darkgrid')

# 创建一个图形和子图
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))  # 两行五列
fig.suptitle('Training and Validation Loss and Metrics')

# 映射列名到绘图的位置
metrics_mapping = {
    'train/box_loss': (0, 0),
    'train/obj_loss': (0, 1),
    'train/cls_loss': (0, 2),
    'metrics/precision': (0, 3),
    'metrics/recall': (0, 4),
    'val/box_loss': (1, 0),
    'val/obj_loss': (1, 1),
    'val/cls_loss': (1, 2),
    'metrics/mAP_0.5': (1, 3),  
    'metrics/mAP_0.5:0.95': (1, 4)  
}

# 绘制每个指标
for metric, pos in metrics_mapping.items():
    ax = axes[pos]
    ax.plot(data[metric], label=f'{metric}', linewidth=2)
    ax.set_title(metric)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric.split('/')[1] if '/' in metric else metric)
    ax.legend()

# 调整布局
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 显示图形
plt.savefig("yolov5m")
