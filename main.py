# Contents of main.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN_GSGD, GSGDOptimizer
from train import train, test
import os

# Data loading, model setup, and main training loop code here
# Define the path where the data should be stored
data_path = './data'

# Check if the data directory exists
download_data = not os.path.exists(os.path.join(data_path, 'MNIST'))

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the datasets, only downloading if necessary
train_dataset = datasets.MNIST(data_path, train=True, download=download_data, transform=transform)
test_dataset = datasets.MNIST(data_path, train=False, download=download_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_GSGD().to(device)
optimizer = GSGDOptimizer(model.parameters(), lr=0.01, rho=10)

for epoch in range(1, 5):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
