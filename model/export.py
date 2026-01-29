import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

CLASSES = 6   # change if your dataset has different count

# SAME architecture as training
model = mobilenet_v2(width_mult=1.0, weights=None)
model.classifier[1] = nn.Linear(model.last_channel, CLASSES)

model.load_state_dict(torch.load("waste_model.pth", map_location="cpu"))
model.eval()

example = torch.rand(1, 3, 160, 160)
scripted = torch.jit.trace(model, example)
scripted.save("waste_model_scripted.pt")

print("Saved waste_model_scripted.pt")
