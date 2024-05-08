"""
1. 先得到至少1k的对抗样本
2. 将对抗样本和原始数据集合并训练新的模型
3. 对比新旧模型在test集上的准确率
4. 在新旧模型上分别进行白盒攻击，对比准确率
5. 在新旧模型上分别进行黑盒攻击，对比准确率
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from task3.model.model import SimpleCNN


def test_model_accuracy_on_test_dataset(model, test_loader):
    # 验证模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def white_attack(model, data):
    pass


def load_black_attack_data(file_name):
    data = np.load(file_name)
    original_images = data['original_images']
    original_labels = data['original_labels']
    perturbed_images = data['perturbed_images']
    perturbed_labels = data['perturbed_labels']
    return original_images, original_labels, perturbed_images, perturbed_labels


def black_attack(model, data):
    model.eval()
    attack_success_list = []

    for original_image, original_label, perturbed_image, perturbed_label in zip(*data):
        with torch.no_grad():
            outputs = model(torch.from_numpy(perturbed_image))
            _, predicted = torch.max(outputs.data, 1)

        if predicted.item() == perturbed_label.item():
            result = (original_image, original_label, perturbed_image, perturbed_label)
            attack_success_list.append(result)

    attack_success_rate = 100 * len(attack_success_list) / len(data[0])
    return attack_success_rate


def test_two_model(func, model_without_attack, model_with_attack, data, desc):
    print(f"\n{'=' * 20}Start test model about {desc}{'=' * 20}")
    model_without_attack_accuracy = func(model_without_attack, data)
    print(f"Model trained without attack dataset about {desc}: {model_without_attack_accuracy}%")

    model_with_attack_accuracy = func(model_with_attack, data)
    print(f"Model trained with attack dataset about {desc}: {model_with_attack_accuracy}%")


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model_without_attack = SimpleCNN()
    model_without_attack.load_state_dict(torch.load("model/model_params_without_attack.pth"))

    model_with_attack = SimpleCNN()
    model_with_attack.load_state_dict(torch.load("model/model_params_with_attack.pth"))

    # 测试两个模型在测试集上的准确率
    test_two_model(test_model_accuracy_on_test_dataset,
                   model_without_attack,
                   model_with_attack,
                   test_loader,
                   "model accuracy on test dataset")

    # 测试两个模型在白盒攻击上的成功率
    pass

    # 测试两个模型在黑盒攻击上的成功率
    black_attack_data = load_black_attack_data("data/black_attack_data_79.npz")
    test_two_model(black_attack,
                   model_without_attack,
                   model_with_attack,
                   black_attack_data,
                   "attack success rate using black attack")
