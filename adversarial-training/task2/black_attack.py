import numpy as np

# 加载数据
data = np.load('../task1/data/successful_samples.npz')
original_images = data['original_images']
original_labels = data['original_labels']
perturbed_images = data['perturbed_images']
perturbed_labels = data['perturbed_labels']
