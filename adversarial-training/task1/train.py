import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from task1.model.model import SimpleCNN


# 模型训练函数
def train_model(epochs, log_step, model, train_loader, device):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    model.train()
    for epoch in range(epochs):  # 训练10个epoch
        print(f"Epoch {epoch + 1}")
        idx = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            idx += 1
            if idx % log_step == 0:
                print(f"    Loss: {loss}")


def test_model(test_loader, device):
    # 验证模型
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


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Start load dataset")
    train_dataset = datasets.FashionMNIST('./data', train=True, download=False, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("Load dataset finish ")

    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(15, 30, model, train_loader, device)
    accuracy = test_model(test_loader, device)
    print(f'Accuracy: {accuracy}%')
    torch.save(model.state_dict(), f'./model/model_params-{accuracy}%.pth')
