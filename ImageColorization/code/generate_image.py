import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import logging 
import matplotlib.pyplot as plt
from utils import send_notice
from torchvision.models import vgg16
import torch.utils.data as data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:feature_layers])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.features(x)
        y_features = self.features(y)
        return nn.functional.mse_loss(x_features, y_features)

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x, y):
        return nn.functional.mse_loss(x, y)

# 自定义数据集类
class ColorizationDataset(Dataset):
    def __init__(self, color_dir, bw_dir, transform=None):
        self.color_dir = color_dir
        self.bw_dir = bw_dir
        self.transform = transform
        self.color_images = os.listdir(color_dir)
        self.bw_images = os.listdir(bw_dir)

    def __len__(self):
        # return 100
        return len(self.color_images)

    def __getitem__(self, idx):
        color_img_path = os.path.join(self.color_dir, self.color_images[idx])
        bw_img_path = os.path.join(self.bw_dir, self.bw_images[idx])
        color_img = Image.open(color_img_path).convert('RGB')
        bw_img = Image.open(bw_img_path).convert('L')

        if self.transform:
            color_img = self.transform(color_img)
            bw_img = self.transform(bw_img)

        return bw_img, color_img

# U-Net模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU()
        )
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        
        self.output = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottom = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottom)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.output(dec1)

if __name__ == "__main__":
    exper_name = "mse"
    dir_name = f'../result/{exper_name}/param'

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    logger.info("Load dataset")

    test_dataset = ColorizationDataset('../dataset/images_test', '../dataset/images_test_black', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    output_result = []
    for i in tqdm(range(10)):
        epoch = (i+1)*5
        model = UNet()
        model.load_state_dict(torch.load(os.path.join(dir_name, f'colorization_unet_{epoch}.pth')))
        model.cuda()
        model.eval()
        
        gray_image, color_image = next(iter(test_loader))
        gray_image = gray_image.cuda()
        color_image = color_image.cuda()
        with torch.no_grad():
            output = model(gray_image)

        gray_image = gray_image.cpu().squeeze(0).squeeze(0)  # 将灰度图像从 (1, 1, H, W) 压缩到 (H, W)
        color_image = color_image.cpu().squeeze(0).squeeze(0)
        output = output.cpu().squeeze(0)
        
        output_result.append(output)

    fig, axes = plt.subplots(10, 2+len(output_result), figsize=(12, 15))
    img_num = 10
    axes[0, 0].set_title('Origin')

    for i in range(len(output_result)):
        axes[0, i+1].set_title(f'Epoch_{5*(i+1)}')
    axes[0, 1+len(output_result)].set_title('Target')

    for i in range(img_num):
        gray_image[i][0] = np.clip(gray_image[i][0], 0, 1)
        axes[i, 0].imshow(gray_image[i][0], cmap='gray')  # 选择第一个灰度图像并显示
        axes[i, 0].axis('off')
        
        for j in range(len(output_result)):
            output = output_result[j]
            output[i] = np.clip(output[i], 0, 1)
            axes[i, j+1].imshow(output[i].permute(1, 2, 0).numpy())
            axes[i, j+1].axis('off')

        color_image[i] = np.clip(color_image[i], 0, 1)
        axes[i, 1+len(output_result)].imshow(color_image[i].permute(1, 2, 0).numpy())
        axes[i, 1+len(output_result)].axis('off')

    plt.savefig(f'comparison_{exper_name}.png')
    