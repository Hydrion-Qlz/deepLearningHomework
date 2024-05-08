"""
1. 先得到至少1k的对抗样本
2. 将对抗样本和原始数据集合并训练新的模型
3. 对比新旧模型在test集上的准确率
4. 在新旧模型上分别进行白盒攻击，对比准确率
5. 在新旧模型上分别进行黑盒攻击，对比准确率
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from task3.model.model import SimpleCNN


def test_model(model, test_loader, device):
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
    test_dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model_without_attack = SimpleCNN()
    model_without_attack.load_state_dict(torch.load("model/model_params_without_attack.pth"))

    model_with_attack = SimpleCNN()
    model_without_attack.load_state_dict(torch.load("model/model_params_with_attack.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_without_attack_accuracy = test_model(model_without_attack, test_loader, device)
    model_with_attack_accuracy = test_model(model_without_attack, test_loader, device)
    print(f"Model without attack dataset accuracy: {model_without_attack_accuracy}")
    print(f"Model with attack dataset accuracy: {model_with_attack_accuracy}")
