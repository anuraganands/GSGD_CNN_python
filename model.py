# Contents of model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# CNN Model Definition
class CNN_GSGD(nn.Module):
    def __init__(self):
        super(CNN_GSGD, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# GSGD Optimizer with Consistency Check
class GSGDOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, rho=10, consistency_threshold=0.1):
        defaults = dict(lr=lr, rho=rho)
        super(GSGDOptimizer, self).__init__(params, defaults)
        self.consistent_batches = []  # Store consistent batches here
        self.consistency_threshold = consistency_threshold  # Define your threshold

    def collect_consistent_batches(self, batch_loss):
        # Check if the batch loss falls within the acceptable range (consistency threshold)
        if len(self.consistent_batches) > 0:
            prev_loss = self.consistent_batches[-1]  # Last consistent batch loss
            if abs(batch_loss - prev_loss) <= self.consistency_threshold:
                self.consistent_batches.append(batch_loss)
            else:
                # If the batch is inconsistent, reset or ignore it based on your strategy
                self.consistent_batches = [batch_loss] if len(self.consistent_batches) >= self.defaults['rho'] else []
        else:
            # Initialize with the first batch loss
            self.consistent_batches.append(batch_loss)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Here we would adjust weights based on consistent batches collected
                if len(self.consistent_batches) >= self.defaults['rho']:
                    # Perform weight update using the consistent batches
                    p.data.add_(d_p, alpha=-group['lr'])
                    self.consistent_batches.clear()  # Reset after weight update with consistent batches
                else:
                    # Regular weight update without consistency filtering
                    ##p.data.add_(-group['lr'], d_p)
                    p.data.add_(d_p, alpha=-group['lr'])


        return loss
