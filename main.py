# Contents of main.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN_GSGD, GSGDOptimizer
from train import train, test

# Data loading, model setup, and main training loop code here
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_GSGD().to(device)
optimizer = GSGDOptimizer(model.parameters(), lr=0.01, rho=10)

for epoch in range(1, 5):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
