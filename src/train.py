import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets import FruitVegDataset  # datasets.py in same folder
from model import get_model           # model.py in same folder
import os

# --- Parameters ---
batch_size = 16
num_epochs = 5
learning_rate = 1e-4

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Dataset ---
# Since we are running from inside 'src', dataset path is one level up
dataset_path = os.path.join(os.getcwd(), "..", "data", "raw")
print("Using dataset path:", dataset_path)

dataset = FruitVegDataset(root_dir=dataset_path, transform=transform)

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

# --- Loss & Optimizer ---
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # shape [batch,1]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.cpu() == labels.cpu()).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Val Accuracy: {val_acc:.4f}")

# --- Save the model ---
results_path = os.path.join(os.getcwd(), "..", "results", "model.pth")
torch.save(model.state_dict(), results_path)
print(f"Model saved to {results_path}")
