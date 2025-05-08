import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json

output_dir = "deepseek_scores.json"
input_dir = "deepseek_prompts_responses.json"

# Make sure this matches the one used during training
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define the same model class as before
class RegressionModel(nn.Module):
    def __init__(self, base_model_name="bert-base-uncased", hidden_size=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, 5)  # 5 output scores
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        return self.regressor(cls_embedding)

# Load model and weights
model = RegressionModel()
model.load_state_dict(torch.load("../Training/results_3/model_epoch_28.pt", map_location="cpu"))
model.eval()

# Labels for the outputs
labels = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

# Load input JSON
with open(input_dir, "r", encoding="utf-8") as infile:
    data = json.load(infile)

# Prepare output
results = []

# Evaluate each prompt-response pair
with torch.no_grad():
    for item in data:
        text = item["prompt"] + "\n" + item["response"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        prediction = model(inputs["input_ids"], inputs["attention_mask"])[0]
        scores = {label: round(value.item(), 2) for label, value in zip(labels, prediction)}
        results.append({
            "prompt": item["prompt"],
            "response": item["response"],
            "scores": scores
        })

# Save to output JSON
with open(output_dir, "w", encoding="utf-8") as outfile:
    json.dump(results, outfile, indent=2, ensure_ascii=False)

print("Saved predictions to ", output_dir)