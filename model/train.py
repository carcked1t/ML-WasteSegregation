import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from dataset import WasteDataset
from utils import get_device, get_transforms

DATA_PATH = "data/raw/train"
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.0003

device = get_device()
print("Using device:", device)

transform = get_transforms()
train_data = WasteDataset(DATA_PATH, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(train_data.classes)
print("Classes:", train_data.classes)

# IMPORTANT: width_mult is fixed to 0.5 everywhere
model = mobilenet_v2(
    width_mult=1.0,
    weights=MobileNet_V2_Weights.DEFAULT   # pretrained backbone
)

model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

# Save fresh model
torch.save(model.state_dict(), "waste_model.pth")
print("Saved waste_model.pth")
