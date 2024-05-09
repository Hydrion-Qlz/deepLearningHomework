import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from task1.attack_dataset import CustomDataset
from task1.model.model import SimpleCNN


def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    images, labels = data[0], data[1]

    images = np.array(images, dtype='float32') / 255.0
    images = images.reshape(-1, 28, 28)
    return images, labels


def get_correct_test_loader(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    correct_images = []
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            if predicted.item() == labels:
                correct += 1
                correct_images.append((images, labels))

    print(f"测试数据集总共{total}张图片, 其中预测正确的图片有{correct}张")
    correct_images = [(image.squeeze(0), label) for image, label in correct_images]
    images, labels = zip(*correct_images)
    np.savez(f"data/attack_image_10k_{len(images)}.npz", images=images, labels=labels)

    correct_dataset = torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))
    correct_loader = DataLoader(correct_dataset, batch_size=1, shuffle=True)
    return correct_loader


def plot_images(original_images, original_labels, perturbed_images, new_labels, file_path):
    plt.figure(figsize=(len(original_images), 2))
    num_images = len(original_images)

    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].squeeze().cpu().detach().numpy(), cmap='gray')
        plt.title(f"Origin: {original_labels[i].item()}")
        plt.axis('off')

        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(perturbed_images[i].squeeze().cpu().detach().numpy(), cmap='gray')
        plt.title(f"Attack: {new_labels[i].item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(file_path)


def save_attack_success_dataset(samples, file_path):
    original_images, original_labels, perturbed_images, perturbed_labels = zip(
        *[(x[0], x[1], x[2], x[3]) for x in samples])

    original_images_np = np.array([img.cpu().detach().numpy() for img in original_images])
    original_labels_np = np.array([lbl.cpu().detach().numpy() for lbl in original_labels])
    perturbed_images_np = np.array([pimg.cpu().detach().numpy() for pimg in perturbed_images])
    perturbed_labels_np = np.array([plbl.cpu().detach().numpy() for plbl in perturbed_labels])

    # 保存到.npy文件
    np.savez(file_path,
             original_images=original_images_np,
             original_labels=original_labels_np,
             perturbed_images=perturbed_images_np,
             perturbed_labels=perturbed_labels_np)


def select_10_samples_and_save(successful_samples, file_path):
    selected_samples = random.sample(successful_samples, min(10, len(successful_samples)))
    original_images, original_labels, perturbed_images, new_labels = zip(
        *[(x[0], x[1], x[2], x[3]) for x in selected_samples])

    plot_images(original_images, original_labels, perturbed_images, new_labels, file_path)


def white_attack(model, data_loader, target_classes, iterations):
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
            # perturbed_data = image.detach().cpu().numpy()
            successful_samples.append((origin_image, label, image, target_label))

        attempts += 1
    return successful_samples, attempts


if __name__ == '__main__':
    dataset_type = "10k"
    show_all_kind = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    model.load_state_dict(torch.load('./model/model_params.pth'))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)
    if dataset_type == "1k":
        images, labels = load_data('data/attack_1k/correct_1k.pkl')
        dataset = CustomDataset(images, labels, transform=transform)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # test_loader = get_correct_test_loader(model, dataset)

    target_classes = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0}
    iterations = 5

    successful_samples, attempts = white_attack(model, data_loader, target_classes, iterations)

    attack_success_rate = 100 * len(successful_samples) / attempts
    print(
        f'Attack Success Rate: {attack_success_rate}%, 总攻击样本数: {attempts}, 总攻击成功数量: {len(successful_samples)}')

    select_10_samples_and_save(successful_samples,
                               f'result/test-{dataset_type}-result/result-iteration-{iterations}-{attack_success_rate}%.png')

    save_attack_success_dataset(successful_samples,
                                f'result/test-{dataset_type}-result/successful_attack_sample-iteration-{iterations}-{len(successful_samples)}.npz')
