# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import os
import json

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# import torchmetrics
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="data/",
    download=True,
    train=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data/",
    download=True,
    train=False,
    transform=transform
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# def imshow(img):
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()

# # get some random training images
# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# labels
# # show images
# imshow(torchvision.utils.make_grid(images))

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    accuracy = float(num_correct) / float(num_samples) * 100
    print(f"Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}")
    model.train()
    return accuracy

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CNN(in_channels=1, num_classes=10).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        loop = tqdm(train_loader, leave=True)

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")


    print("Checking accuracy on training set")
    train_acc = check_accuracy(train_loader, model)

    print("Checking accuracy on test data...")
    test_acc = check_accuracy(test_loader, model)

    model_path = "models/baseline_model.pt"
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")


    #Add basic metrics

    model_size_mb = os.path.getsize("models/baseline_model.pt") / (1024 * 1024)
    print(f"Model Size: {model_size_mb:.2f} MB")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n===== METRICS SUMMARY =====")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy:  {test_acc:.2f}%")
    print(f"Model Size:     {model_size_mb:.2f} MB")
    print(f"Parameters:     {num_params}")

    results = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "model_size_mb": model_size_mb,
        "parameters": num_params,
        "epochs": epochs,
        "batch_size": batch_size
    }

    results_path = "results/results_baseline.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")