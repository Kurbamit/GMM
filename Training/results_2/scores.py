import torch
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import pearsonr
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# Dataset definition
# -----------------------------
class HelpSteerDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=512):
        self.samples = []
        with open(filepath, 'r') as f:
            for line in f:
                item = json.loads(line)
                text = item["prompt"] + "\n" + item["response"]
                scores = [item["helpfulness"], item["correctness"], item["coherence"], item["complexity"], item["verbosity"]]
                self.samples.append((text, torch.tensor(scores, dtype=torch.float)))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, scores = self.samples[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': scores
        }

# -----------------------------
# Model definition
# -----------------------------
class RegressionModel(nn.Module):
    def __init__(self, base_model_name="bert-base-uncased", hidden_size=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)  # 5 output scores
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        return self.regressor(cls_embedding)

# -----------------------------
# Prediction and evaluation
# -----------------------------
def get_predictions(model, dataloader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask).cpu()
            all_preds.append(outputs)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).int()
    all_labels = torch.cat(all_labels).int()
    return all_labels, all_preds

def evaluate(y_true, y_pred):
    y_pred_rounded = torch.clamp(torch.round(y_pred), 0, 4).int()
    y_true = y_true.int()

    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    y_pred_rounded_np = y_pred_rounded.numpy()

    names = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    results = {}

    for i, name in enumerate(names):
        true_vals = y_true_np[:, i]
        pred_vals = y_pred_np[:, i]
        pred_rounded = y_pred_rounded_np[:, i]

        results[name] = {
            "MAE (raw)": mean_absolute_error(true_vals, pred_vals),
            "MAE (rounded)": mean_absolute_error(true_vals, pred_rounded),
            "Accuracy": accuracy_score(true_vals, pred_rounded),
            "Pearson": pearsonr(true_vals, pred_vals)[0]
        }

    return results

# -----------------------------
# Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

val_dataset = HelpSteerDataset("validation.jsonl", tokenizer)
val_loader = DataLoader(val_dataset, batch_size=16)

model = RegressionModel()
model.load_state_dict(torch.load("model_epoch_28.pt", map_location=device))
model.to(device)

# -----------------------------
# Run evaluation
# -----------------------------
y_true, y_pred = get_predictions(model, val_loader, device)
metrics = evaluate(y_true, y_pred)

for category, scores in metrics.items():
    print(f"\n{category.upper()}")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

# HELPFULNESS
#   MAE (raw): 1.0400
#   MAE (rounded): 1.0400
#   Accuracy: 0.3300
#   Pearson: 0.2930

# CORRECTNESS
#   MAE (raw): 1.0900
#   MAE (rounded): 1.0900
#   Accuracy: 0.2800
#   Pearson: 0.2786

# COHERENCE
#   MAE (raw): 0.8900
#   MAE (rounded): 0.8900
#   Accuracy: 0.2300
#   Pearson: 0.2187

# COMPLEXITY
#   MAE (raw): 0.7700
#   MAE (rounded): 0.7700
#   Accuracy: 0.3200
#   Pearson: 0.3795

# VERBOSITY
#   MAE (raw): 0.5700
#   MAE (rounded): 0.5700
#   Accuracy: 0.4900
#   Pearson: 0.4354