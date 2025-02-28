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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 自定义数据集类
class ColorizationDataset(Dataset):
    def __init__(self, color_dir, bw_dir, transform=None):
        self.color_dir = color_dir
        self.bw_dir = bw_dir
        self.transform = transform
        self.color_images = os.listdir(color_dir)
        self.bw_images = os.listdir(bw_dir)

    def __len__(self):
        return 10000
        # return len(self.color_images)

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

def denormalize(img):
    img = img * 0.5 + 0.5
    img = img.clamp(0, 1)
    return img


dir_name = ".."
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# todo: train-val split
logger.info("Start loading dataset")
train_dataset = ColorizationDataset(f'{dir_name}/dataset/images_train', f'{dir_name}/dataset/images_train_black', transform=transform)
test_dataset = ColorizationDataset(f'{dir_name}/dataset/images_test', f'{dir_name}/dataset/images_test_black', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
logger.info("Loading dataset finish")

save_folder = f'{dir_name}/result/run1'
logger.info("Start loading model")
model = UNet().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
logger.info("Loading model finish")

for idx in range(10):
    outputs = []
    # 加载模型
    for i in range(idx+1):
        epoch = (i+1)*5
        model = UNet()
        model.load_state_dict(torch.load(f'../result/run1/colorization_unet_{epoch}.pth'))
        model.cuda()
        model.eval()

        gray_image, color_image = next(iter(val_loader))
        gray_image = gray_image.cuda()
        color_image = color_image.cuda()
        with torch.no_grad():
            output = model(gray_image)
        
        gray_image = gray_image.cpu().squeeze(0).squeeze(0)  # 将灰度图像从 (1, 1, H, W) 压缩到 (H, W)
        color_image = color_image.cpu().squeeze(0).squeeze(0)
        output = denormalize(output.cpu().squeeze(0))
        outputs.append(output)

    fig, axes = plt.subplots(10, 2+len(outputs), figsize=(12, 15))

    img_num = 10
    axes[0, 0].set_title('Origin')
    for i in range(len(outputs)):
        axes[0, i+1].set_title(f'Epoch_{5*(i+1)}')
    axes[0, 1+len(outputs)].set_title('Target')

    for i in range(img_num):
        axes[i, 0].imshow(gray_image[i][0], cmap='gray')  # 选择第一个灰度图像并显示
        axes[i, 0].axis('off')
        
        for j in range(len(outputs)):
            output = outputs[j]
            axes[i, j+1].imshow(output[i].permute(1, 2, 0).numpy())
            axes[i, j+1].axis('off')

        axes[i, 1+len(outputs)].imshow(color_image[i].permute(1, 2, 0).numpy())
        axes[i, 1+len(outputs)].axis('off')

    plt.savefig(f'{save_folder}/comparison_{epoch}.png')
    plt.show()