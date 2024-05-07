import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from task1.model.model import SimpleCNN


def test_model(test_loader, device):
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
    model = SimpleCNN()  # 创建模型实例
    model.load_state_dict(torch.load('./model/model_params.pth'))  # 加载保存的模型参数

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = test_model(test_loader, device)
    print(f"Model Accuracy on test dataset: {accuracy}%")  # 91.56%
