import torch
import torch.nn as nn
import json
import os
from baseline_mnist import BaselineCNN, get_dataloaders, evaluate
from torchao.quantization import quantize_, int8_weight_only

DEVICE = torch.device("cpu")

model = BaselineCNN()
if not os.path.exists("models/baseline_model.pt"):
    print("Error: baseline_model.pt not found. Run baseline_mnist.py first.")
else:
    model.load_state_dict(torch.load("models/baseline_model.pt", map_location=DEVICE))
model.eval()


quantize_(model, int8_weight_only())

_, test_loader = get_dataloaders(batch_size=64)

criterion = nn.CrossEntropyLoss()
_, acc = evaluate(model, test_loader, criterion, DEVICE)
print(f"Quantized (torchao) model accuracy: {acc:.2f}%")

torch.save(model.state_dict(), "models/quantized_model.pt")
print("Quantized model saved to models/quantized_model.pt")