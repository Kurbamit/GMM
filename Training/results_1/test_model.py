import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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
            nn.Linear(256, 5)  # 5 output scores
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        return self.regressor(cls_embedding)

# Load model and weights
model = RegressionModel()
model.load_state_dict(torch.load("model_epoch_20.pt", map_location="cpu"))
model.eval()

############################################################
# Example input
prompt = "What is the capital of France?"
response = "The capital of France is Paris. It is known for its art, fashion, and culture. The Eiffel Tower is one of its most famous landmarks."

# Combine and tokenize
text = prompt + "\n" + response
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Run model
with torch.no_grad():
    prediction = model(inputs["input_ids"], inputs["attention_mask"])

print("Model prediction:")
print(prediction)

# Print results
labels = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
for name, value in zip(labels, prediction[0]):
    print(f"{name}: {value.item():.2f}")
