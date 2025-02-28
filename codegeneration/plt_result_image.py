import re
import matplotlib.pyplot as plt

# 读取文件
with open('log/all.log', 'r') as file:
    lines = file.readlines()

# 处理数据
epochs = []
train_loss = []
valid_loss = []
train_em = []
valid_em = []
train_bleu = []
valid_bleu = []

i = 1
for line in lines:
    if line.startswith('Epoch'):
        epoch_match = re.search(r'Epoch (\d+)', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            epochs.append(i)
            i += 1

        loss_match = re.search(r'Training Loss: (\d+\.\d+)', line)
        if loss_match:
            train_loss.append(float(loss_match.group(1)))

        loss_match = re.search(r'Validation Loss: (\d+\.\d+)', line)
        if loss_match:
            valid_loss.append(float(loss_match.group(1)))

        em_match = re.search(r'Training Exact Match: (\d+\.\d+)', line)
        if em_match:
            train_em.append(float(em_match.group(1)))

        em_match = re.search(r'Validation Exact Match: (\d+\.\d+)', line)
        if em_match:
            valid_em.append(float(em_match.group(1)))

        bleu_match = re.search(r'Training BLEU: (\d+\.\d+)', line)
        if bleu_match:
            train_bleu.append(float(bleu_match.group(1)))

        bleu_match = re.search(r'Validation BLEU: (\d+\.\d+)', line)
        if bleu_match:
            valid_bleu.append(float(bleu_match.group(1)))

# 绘制学习曲线
plt.figure(figsize=(12, 8))

# Loss
plt.subplot(2, 1, 1)
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, valid_loss, 'r-', label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Exact Match and BLEU Score
plt.subplot(2, 1, 2)
plt.plot(epochs, train_em, 'b--', label='Training Exact Match')
plt.plot(epochs, valid_em, 'r--', label='Validation Exact Match')
plt.plot(epochs, train_bleu, 'b-', label='Training BLEU')
plt.plot(epochs, valid_bleu, 'r-', label='Validation BLEU')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
file_name = f'./result/all.png'
plt.savefig(file_name)
plt.show()