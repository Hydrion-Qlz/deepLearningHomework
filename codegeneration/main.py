import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_scheduler
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
from utils import send_notice
import traceback
import datetime

class TextCodeDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.data[idx]['nl'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        targets = self.tokenizer(self.data[idx]['code'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze(), 'labels': targets['input_ids'].squeeze()}

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['DEVICE_EVALUATE'] = "cuda:0"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    dataset = load_dataset('./dataset')
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']
    test_dataset = dataset['test']

    # model_path = './hf_hub/models--google-t5--t5-small'
    model_path = './seq2seq_model'
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    train_dataset = TextCodeDataset(tokenizer, train_dataset)
    valid_dataset = TextCodeDataset(tokenizer, valid_dataset)
    valid_dataset = TextCodeDataset(tokenizer, test_dataset)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=48, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=48, sampler=valid_sampler)

    optimizer = AdamW(model.parameters(), lr=4e-5)
    num_epochs = 30
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device(f'cuda:{rank}')
    model.to(device)
    model = DDP(model, device_ids=[rank])

    train_losses = []
    valid_losses = []
    train_exact_matches = []
    valid_exact_matches = []
    train_bleu_scores = []
    valid_bleu_scores = []

    model.train()
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        train_loss = 0
        train_exact_match = 0
        train_bleu_score = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            train_loss += loss.item()

            with torch.no_grad():
                generated_ids = model.module.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=128)
                decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                
                for pred, label in zip(decoded_preds, decoded_labels):
                    if pred == label:
                        train_exact_match += 1
                    train_bleu_score += sentence_bleu([label.split()], pred.split(), smoothing_function=SmoothingFunction().method1)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_exact_matches.append(train_exact_match / len(train_loader.dataset))
        train_bleu_scores.append(train_bleu_score / len(train_loader.dataset))

        model.eval()
        valid_loss = 0
        valid_exact_match = 0
        valid_bleu_score = 0

        for batch in valid_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                valid_loss += outputs.loss.item()

                generated_ids = model.module.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=128)
                decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                
                for pred, label in zip(decoded_preds, decoded_labels):
                    if pred == label:
                        valid_exact_match += 1
                    valid_bleu_score += sentence_bleu([label.split()], pred.split(), smoothing_function=SmoothingFunction().method1)

        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        valid_exact_matches.append(valid_exact_match / len(valid_loader.dataset))
        valid_bleu_scores.append(valid_bleu_score / len(valid_loader.dataset))

        print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}, Training Exact Match: {train_exact_matches[-1]}, Validation Exact Match: {valid_exact_matches[-1]}, Training BLEU: {train_bleu_scores[-1]}, Validation BLEU: {valid_bleu_scores[-1]}')
        model.train()

        if rank == 0:
            model.module.save_pretrained('./seq2seq_model')
            tokenizer.save_pretrained('./seq2seq_model')

            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(range(1, epoch + 2), train_losses, 'b-', label='Training Loss')
            plt.plot(range(1, epoch + 2), valid_losses, 'r-', label='Validation Loss')
            plt.title('Learning Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(range(1, epoch + 2), train_exact_matches, 'b--', label='Training Exact Match')
            plt.plot(range(1, epoch + 2), valid_exact_matches, 'r--', label='Validation Exact Match')
            plt.plot(range(1, epoch + 2), train_bleu_scores, 'b-', label='Training BLEU')
            plt.plot(range(1, epoch + 2), valid_bleu_scores, 'r-', label='Validation BLEU')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()

            plt.tight_layout()
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            file_name = f'./result/learning_curve_{current_time}_Epoch_{epoch+1}.png'
            plt.savefig(file_name)
            plt.show()
            send_notice("训练结束, 训练结果保存于" + file_name) 

    cleanup()


def main():
    try:
        world_size = 3
        mp.spawn(train,
                args=(world_size,),
                nprocs=world_size,
                join=True)
    except:
        print(traceback.format_exc())
        send_notice("训练结果异常\n\n" + traceback.format_exc())

if __name__ == "__main__":
    main()
