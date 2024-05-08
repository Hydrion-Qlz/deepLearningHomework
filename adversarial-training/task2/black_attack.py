import numpy as np
import torch

from task2.model.model import CNN


def load_data(file_name):
    # 加载数据
    data = np.load(file_name)
    original_images = data['original_images']
    original_labels = data['original_labels']
    perturbed_images = data['perturbed_images']
    perturbed_labels = data['perturbed_labels']
    return original_images, original_labels, perturbed_images, perturbed_labels


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
    print(len(attack_success_list))
    print(len(data[0]))
    print(f"Success Rate: {len(attack_success_list) / len(data[0]) * 100}%")
