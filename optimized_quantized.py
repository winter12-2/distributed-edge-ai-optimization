import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import json
import os

# using cpu for quantization
DEVICE = torch.device("cpu")

# create folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# same cnn model from baseline
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# function to test the model
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# load test data
transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# create baseline model structure
model = CNN()

# check if baseline model exists
if not os.path.exists("models/baseline_model.pt"):
    print("Error: baseline_model.pt not found. Run baseline_mnist.py first.")
    exit()

# load trained baseline weights
model.load_state_dict(torch.load("models/baseline_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Baseline model loaded successfully.")

# count parameters before quantization
total_params = sum(p.numel() for p in model.parameters())

# apply dynamic quantization
# this mainly quantizes the linear layer
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

print("Dynamic quantization applied.")

# test quantized model
criterion = nn.CrossEntropyLoss()
test_loss, test_acc = evaluate(quantized_model, test_loader, criterion, DEVICE)

print(f"Quantized Model Test Accuracy: {test_acc:.2f}%")

# save quantized model
quantized_model_path = "models/quantized_model.pt"
torch.save(quantized_model.state_dict(), quantized_model_path)

# calculate model size
model_size_mb = os.path.getsize(quantized_model_path) / (1024 * 1024)

# save results
results = {
    "model_name": "Quantized Optimized MNIST CNN",
    "quantization_type": "PyTorch dynamic int8 quantization",
    "test_accuracy": round(test_acc, 2),
    "test_loss": round(test_loss, 4),
    "model_size_mb": round(model_size_mb, 4),
    "parameters": total_params
}

with open("results/results_quantized.json", "w") as f:
    json.dump(results, f, indent=4)

print("Quantized model saved to models/quantized_model.pt")
print("Results saved to results/results_quantized.json")

print("\n===== QUANTIZED MODEL SUMMARY =====")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Model Size:    {model_size_mb:.4f} MB")
print(f"Parameters:    {total_params}")