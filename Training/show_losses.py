import json
import matplotlib.pyplot as plt

# Load loss values from JSON files
with open('./results_3/train_losses.json', 'r') as f:
    train_losses = json.load(f)

with open('./results_3/val_losses.json', 'r') as f:
    val_losses = json.load(f)

# Check that both lists have the same length
assert len(train_losses) == len(val_losses), "Mismatch in number of epochs between train and validation losses"

# Create epoch range
epochs = range(1, len(train_losses) + 1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch. Result 1')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
