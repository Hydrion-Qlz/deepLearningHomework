import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from task1.model.model import SimpleCNN


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


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

    correct_dataset = torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))
    correct_loader = DataLoader(correct_dataset, batch_size=1, shuffle=True)
    return correct_loader


def plot_images(original_images, original_labels, perturbed_images, new_labels, file_path):
    plt.figure(figsize=(10, 2))
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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    model.load_state_dict(torch.load('./model/model_params.pth'))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)
    test_loader = get_correct_test_loader(model, test_dataset)

    # 定义目标类别映射
    target_classes = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0}

    # 准备模型
    model.eval()

    # 攻击成功的计数器
    successful_samples = []
    label_not_7 = 0
    attempts = 0

    for images, labels in test_loader:
        # if attempts >= 1000 or len(successful_samples) > 20:
        #     break
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        # 设置目标类别
        target_label = torch.tensor([target_classes[labels.item()]], device=device)

        # 只对原始分类正确的图像进行攻击
        output = model(images)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != labels.item():
            continue

        # 计算损失
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target_label)
        model.zero_grad()
        loss.backward()
        data_grad = images.grad.data

        # 调用FGSM攻击
        epsilon = 0.0001  # 攻击强度
        perturbed_data = fgsm_attack(images, epsilon, data_grad)

        # 重新分类扰动后的图像
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        # if final_pred.item() == target_label.item() and labels != 7:
        if final_pred.item() == target_label.item():
            if labels != 7:
                label_not_7 += 1
            successful_samples.append((images, labels, perturbed_data, final_pred))

        attempts += 1

    attack_success_rate = 100 * len(successful_samples) / attempts
    print(
        f'Attack Success Rate: {attack_success_rate}%, 不是第七类的数量: {label_not_7}, 总攻击成功数量: {len(successful_samples)}')

    selected_samples = random.sample(successful_samples, min(10, len(successful_samples)))
    original_images, original_labels, perturbed_images, new_labels = zip(
        *[(x[0], x[1], x[2], x[3]) for x in selected_samples])

    # 绘制并保存图像
    plot_images(original_images, original_labels, perturbed_images, new_labels, f'data/result-{epsilon}.png')
