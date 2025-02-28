from train_UNet import *
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

def calculate_color_difference(true_image, generated_image):
    true_image_lab = cv2.cvtColor(true_image, cv2.COLOR_RGB2LAB)
    generated_image_lab = cv2.cvtColor(generated_image, cv2.COLOR_RGB2LAB)
    color_difference = np.mean((true_image_lab - generated_image_lab) ** 2)
    return color_difference

def calculate_ssim(true_image, generated_image):
    # 设置窗口大小为图像最小边长，并确保是奇数
    win_size = min(true_image.shape[0], true_image.shape[1], 7)
    win_size = win_size if win_size % 2 == 1 else win_size - 1
    return ssim(true_image, generated_image, multichannel=True, win_size=win_size)

def calculate_psnr(true_image, generated_image):
    mse = np.mean((true_image - generated_image) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def PSNR():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation PSNR Over Epochs')
    plt.savefig(f'{save_folder}/loss_{epoch+1}.png')
    plt.show()


if __name__ == "__main__":  
    exper_name = "mse+vgg+l1+l2_1e-5"
    dir_name = f'../result/{exper_name}/param'
    train_psnr_values = []
    train_ssim_values = []
    
    val_psnr_values = []
    val_ssim_values = []

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    logger.info("Load dataset")
    train_dataset = ColorizationDataset('../dataset/images_train', '../dataset/images_train_black', transform=transform)
    data_size = len(train_dataset)
    train_ratio = 0.9
    train_size, val_size = int(round(data_size*train_ratio)), int(round(data_size*(1-train_ratio)))
    train_dataset, val_dataset = data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    for i in range(10):
        epoch = (i+1)*5
        model = UNet()
        model.load_state_dict(torch.load(os.path.join(dir_name, f'colorization_unet_{epoch}.pth')))
        model.cuda()
        model.eval()
        
        train_psnr = []
        train_ssim = []
        val_psnr = []
        val_ssim = []

        with torch.no_grad():
            for gray_images, color_images in tqdm(train_loader, desc=f"Epoch: {epoch}/50: train dataset"):
                gray_images = gray_images.cuda()
                color_images = color_images.cuda()

                output = model(gray_images).cpu().numpy()
                color_images = color_images.cpu().numpy()
                for j in range(len(output)):
                    train_psnr.append(psnr(color_images[j], output[j], data_range=1.0))
                    train_ssim.append(ssim(color_images[j].transpose(1,2,0), output[j].transpose(1,2,0),
                                        multichannel=True,
                                        channel_axis=2,
                                        win_size=31,
                                        data_range=1.0))

            for gray_images, color_images in tqdm(val_loader, desc=f"Epoch: {epoch}/50: valid dataset"):
                gray_images = gray_images.cuda()
                color_images = color_images.cuda()

                output = model(gray_images).cpu().numpy()
                color_images = color_images.cpu().numpy()
                for j in range(len(output)):
                    val_psnr.append(psnr(color_images[j], output[j], data_range=1.0))
                    val_ssim.append(ssim(color_images[j].transpose(1,2,0), output[j].transpose(1,2,0),
                                        multichannel=True,
                                        channel_axis=2,
                                        win_size=31,
                                        data_range=1.0))

        train_psnr_values.append(sum(train_psnr) / len(train_psnr))
        train_ssim_values.append(sum(train_ssim) / len(train_ssim))
        val_psnr_values.append(sum(val_psnr) / len(val_psnr))
        val_ssim_values.append(sum(val_ssim) / len(val_ssim))

    # 绘制结果
    epochs = range(5, (len(train_psnr_values) + 1) * 5, 5)

    plt.figure(figsize=(12, 6))

    # 绘制PSNR曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_psnr_values, label='Train PSNR')
    plt.plot(epochs, val_psnr_values, label='Val PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    plt.title('PSNR over Epochs')
    
    # 绘制SSIM曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_ssim_values, label='Train SSIM')
    plt.plot(epochs, val_ssim_values, label='Val SSIM')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.title('SSIM over Epochs')

    plt.tight_layout()
    plt.show()
    plt.savefig('psnr_plot_mse.png')


