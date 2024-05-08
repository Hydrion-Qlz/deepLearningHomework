import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from task2.model.model import CNN


def load_data(file_name):
    # 加载数据
    data = np.load(file_name)
    original_images = data['original_images']
    original_labels = data['original_labels']
    perturbed_images = data['perturbed_images']
    perturbed_labels = data['perturbed_labels']
    return original_images, original_labels, perturbed_images, perturbed_labels


def plot_images(original_images, original_labels, perturbed_images, new_labels, file_path):
    plt.figure(figsize=(len(original_images), 2))
    num_images = len(original_images)

    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Origin: {original_labels[i].item()}")
        plt.axis('off')

        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(perturbed_images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Attack: {new_labels[i].item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(file_path)


def black_attack(black_model, data):
    black_model.eval()
    attack_success_list = []

    for original_image, original_label, perturbed_image, perturbed_label in zip(*data):
        with torch.no_grad():
            outputs = black_model(torch.from_numpy(perturbed_image))
            _, predicted = torch.max(outputs.data, 1)

        if predicted.item() == perturbed_label.item():
            result = (original_image, original_label, perturbed_image, perturbed_label)
            attack_success_list.append(result)

    return attack_success_list


if __name__ == '__main__':
    data = load_data('data/successful_attack_samples-0.001-79.npz')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    black_model = CNN().to(device)
    black_model.load_state_dict(torch.load("model/cnn.ckpt"))

    attack_success_list = black_attack(black_model, data)
    success_rate = len(attack_success_list) / len(data[0])
    print(f"Dataset Num: {len(data[0])}")
    print(f"Attack Success Num: {len(attack_success_list)}")
    print(f"Success Rate: {success_rate * 100}%")

    # samples = [sample for sample in attack_success_list if sample[1] != 7]
    selected_samples = random.sample(attack_success_list, min(10, len(samples)))
    original_images, original_labels, perturbed_images, new_labels = zip(
        *[(x[0], x[1], x[2], x[3]) for x in selected_samples])
    plot_images(original_images, original_labels, perturbed_images, new_labels,
                f"result/black-attack-result-{success_rate * 100}%.png")
