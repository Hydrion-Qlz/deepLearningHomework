import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# 定义数据集类（与训练脚本中的相同）
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


def compute(model, loader, desc):
    predictions = []
    references = []
    exact_matches = 0
    total_samples = 0
    bleu_scores = []

    for batch in tqdm(loader, desc=f"Evaluating on {desc}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions.extend(decoded_preds)
        references.extend(decoded_labels)

        for pred, label in zip(decoded_preds, decoded_labels):
            if pred == label:
                exact_matches += 1
            total_samples += 1
            bleu_scores.append(sentence_bleu([label.split()], pred.split(), smoothing_function=SmoothingFunction().method1))

    exact_match_score = exact_matches / total_samples
    bleu_score = sum(bleu_scores) / len(bleu_scores)
    return exact_match_score, bleu_score


def evaluate(model_path, desc):
    print(f"{'='*30} Evaluate on {desc} {'='*30}")
    # 加载训练好的模型
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    model.to(device)

    # 评估验证集
    valid_exact_match, valid_bleu = compute(model, valid_loader, desc)
    print(f'Validation Exact Match: {valid_exact_match:.4f}, Validation BLEU: {valid_bleu:.4f}')

    # 评估测试集
    test_exact_match, test_bleu = compute(model, test_loader, desc)
    print(f'Test Exact Match: {test_exact_match:.4f}, Test BLEU: {test_bleu:.4f}')



if __name__ == "__main__":
    # 加载数据集
    dataset = load_dataset('./dataset')
    test_dataset = dataset['test']
    valid_dataset = dataset['validation']
    tokenizer = T5Tokenizer.from_pretrained('./seq2seq_model')

    test_dataset = TextCodeDataset(tokenizer, test_dataset)
    valid_dataset = TextCodeDataset(tokenizer, valid_dataset)

    test_loader = DataLoader(test_dataset, batch_size=8)
    valid_loader = DataLoader(valid_dataset, batch_size=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluate("./hf_hub/models--google-t5--t5-small", "Base Model")
    evaluate("./seq2seq_model", "Finetune Model")
