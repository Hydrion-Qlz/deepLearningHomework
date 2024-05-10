"""
1. 先得到至少1k的对抗样本
2. 将对抗样本和原始数据集合并训练新的模型
3. 对比新旧模型在test集上的准确率
4. 在新旧模型上分别进行白盒攻击，对比准确率
5. 在新旧模型上分别进行黑盒攻击，对比准确率
"""
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from task3.model.custom_dataset import CustomDataset
from task3.model.model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model_accuracy_on_test_dataset(model, test_loader):
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


def load_white_attack_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)


def white_attack(model, data_loader):
    target_classes = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0}
    iterations = 15
    model.eval()
    successful_samples = []
    attempts = 0
    criterion = nn.CrossEntropyLoss()
    for image, label in tqdm(data_loader, desc="white attack process"):
        origin_image = image.clone()
        image, label = image.to(device), label.to(device)
        image.requires_grad = True

        target_label = torch.tensor([target_classes[label.item()]], device=device)

        # 只对原始分类正确的图像进行攻击
        output = model(image)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != label.item():
            continue

        # 白盒攻击
        optimizer = torch.optim.Adam([image], lr=0.01)
        attack_success = False
        for _ in range(iterations):
            optimizer.zero_grad()
            output = model(image)
            pred = output.max(1, keepdim=True)[1]
            if pred.item() == target_label.item():
                attack_success = True
                break
            loss = criterion(output, target_label)
            loss.backward()
            optimizer.step()

        # 判断是否攻击成功
        if attack_success:
            successful_samples.append((origin_image, label, image, target_label))

        attempts += 1
    return successful_samples, attempts


def get_white_attack_success_rate(model, data_loader):
    successful_samples, attempts = white_attack(model, data_loader)
    attack_success_rate = 100 * len(successful_samples) / attempts
    return attack_success_rate


def load_black_attack_data(file_name):
    # load 1k dataset
    with open('data/correct_1k.pkl', 'rb') as file:
        data = pickle.load(file)
    images, labels = data[0], data[1]

    images = np.array(images, dtype='float32') / 255.0
    images = images.reshape(-1, 28, 28)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = CustomDataset(images, labels, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return data_loader
    # data = np.load(file_name)
    # original_images = data['original_images']
    # original_labels = data['original_labels']
    # perturbed_images = data['perturbed_images']
    # perturbed_labels = data['perturbed_labels']
    # return original_images, original_labels, perturbed_images, perturbed_labels


def black_attack(model, data):
    model.eval()
    attack_success_list = []

    for original_image, original_label, perturbed_image, perturbed_label in data:
        with torch.no_grad():
            outputs = model(perturbed_image)
            _, predicted = torch.max(outputs.data, 1)

        if predicted.item() == perturbed_label.item():
            result = (original_image, original_label, perturbed_image, perturbed_label)
            attack_success_list.append(result)
    return attack_success_list


def get_black_attack_rate(model, data):
    # get success white attack sample
    successful_white_attack_samples, _ = white_attack(model, data)

    # black attack
    attack_success_list = black_attack(model, successful_white_attack_samples)
    attack_success_rate = len(attack_success_list) / len(data) * 100
    return attack_success_rate


def test_two_model(func, model_without_attack, model_with_attack, data, desc):
    print(f"\n{'=' * 20} Start test model about {desc} {'=' * 20}")
    model_without_attack_accuracy = func(model_without_attack, data)
    print(f"\nModel trained without attack dataset about {desc}: {model_without_attack_accuracy}%")

    model_with_attack_accuracy = func(model_with_attack, data)
    print(f"Model trained with attack dataset about {desc}: {model_with_attack_accuracy}%")


def load_test_accuracy_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader


if __name__ == '__main__':
    model_without_attack = SimpleCNN().to(device)
    model_without_attack.load_state_dict(torch.load("model/model_params_without_attack_91.56%.pth"))

    model_with_attack = SimpleCNN().to(device)
    model_with_attack.load_state_dict(torch.load("model/model_params_with_attack_88.91%.pth"))

    # 测试两个模型在测试集上的准确率
    test_accuracy_data = load_test_accuracy_data()
    test_two_model(test_model_accuracy_on_test_dataset,
                   model_without_attack,
                   model_with_attack,
                   test_accuracy_data,
                   "model accuracy on test dataset")

    # 测试两个模型在白盒攻击上的成功率
    white_attack_data = load_white_attack_data()
    test_two_model(get_white_attack_success_rate,
                   model_with_attack,
                   model_without_attack,
                   white_attack_data,
                   "attack success rate using white attack")

    # 测试两个模型在黑盒攻击上的成功率
    black_attack_data = load_black_attack_data("data/black_attack_data_896.npz")
    test_two_model(get_black_attack_rate,
                   model_with_attack,
                   model_without_attack,
                   black_attack_data,
                   "attack success rate using black attack")
