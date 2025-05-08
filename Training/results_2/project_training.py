import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import json
import os
import matplotlib.pyplot as plt

weights_path = './weights'
loss_path = './losses'
os.makedirs(weights_path, exist_ok=True)
os.makedirs(loss_path, exist_ok=True)
os.path.isdir(weights_path)
os.path.isdir(loss_path)

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
    
def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=5):
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1} | Validation Loss: {avg_val_loss:.4f}")

        if epoch % 3 == 0:
            checkpoint_file = os.path.join(weights_path, f"model_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_file)
            print(f"✅ Saved model to {checkpoint_file}")

        with open(os.path.join(loss_path, "train_losses.json"), "w") as f:
            json.dump(train_losses, f)
        with open(os.path.join(loss_path, "val_losses.json"), "w") as f:
            json.dump(val_losses, f)

    return train_losses, val_losses

def show_train_validation_loss(train_loss, validation_loss):
  plt.plot(train_losses, label="Train Loss")
  plt.plot(validation_loss, label="Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training & Validation Loss")
  plt.legend()
  plt.grid(True)
  plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_dataset = HelpSteerDataset("train.jsonl", tokenizer)
val_dataset = HelpSteerDataset("validation.jsonl", tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16) 

model = RegressionModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

train_losses, validation_losses = train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=30)

show_train_validation_loss(train_losses, validation_losses)
