import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm
import random

from task3.model.custom_dataset import CustomDataset
from task3.model.model import SimpleCNN


def get_correct_predict_data(model, data_num):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST('./data', train=True, download=False, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    correct_images = []
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            if predicted.item() == labels:
                correct += 1
                correct_images.append((images, labels))

    print(f"训练数据集总共{total}张图片, 其中预测正确的图片有{correct}张")
    correct_images = [(image.squeeze(0), label) for image, label in correct_images]
    images, labels = zip(*correct_images)

    selected_indices = random.sample(range(len(images)), data_num)

    selected_images = [images[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]

    return selected_images, selected_labels


def white_attack(model, data_loader, data_num):
    target_classes = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0}
    iterations = 20
    model.eval()
    successful_samples = []
    criterion = nn.CrossEntropyLoss()
    for image, label in tqdm(data_loader, desc="white attack process"):
        if len(successful_samples) >= data_num:
            break
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

        if attack_success:
            successful_samples.append((origin_image, label, image, target_label))

    return successful_samples


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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("model/model_params.pth"))

    dataset = datasets.FashionMNIST('./data', train=True, download=False, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)  # 打乱数据
    data_num = 5000
    samples = white_attack(model, data_loader, data_num)
    save_attack_success_dataset(samples, f"data/attack_image_{min(data_num, len(samples))}.npz")
