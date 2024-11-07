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
    def __init__(self, params, lr=0.01, rho=10, revisit_batch_num=2):
        defaults = dict(lr=lr, rho=rho)
        super(GSGDOptimizer, self).__init__(params, defaults)
        self.consistent_batches = []  # Store consistent data instances
        self.rho = rho

    def collect_consistent_batches(self, batch_loss, data, target, avg_dummy_verification_error):
        # Check consistency based on dummy verification error
        if len(self.consistent_batches) == 0 or batch_loss <= avg_dummy_verification_error + self.rho:
            self.consistent_batches.append((data, target))

    def step(self, model, loss_fn):
        # Only proceed with updating weights if consistent batches exist
        if self.consistent_batches:
            model.train()  # Ensure model is in training mode
            for data, target in self.consistent_batches:
                model.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                
                # Apply gradients to parameters
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.data.add_(p.grad, alpha=-group['lr'])
            # Clear consistent batches after weight update
            self.consistent_batches.clear()



# class GSGDOptimizer(optim.Optimizer):
#     def __init__(self, params, lr=0.01, rho=10, revisit_batch_num=4):
#         defaults = dict(lr=lr, rho=rho)
#         super(GSGDOptimizer, self).__init__(params, defaults)
#         self.consistent_batches = []  # Store consistent data instances
#         self.rho = rho

#     def collect_consistent_batches(self, model, batch_loss, data, target, verification_loader):
#         # Calculate average error on verification data
#         model.eval()  # Set model to evaluation mode
#         with torch.no_grad():
#             total_error = 0
#             count = 0
#             for v_data, v_target in verification_loader:
#                 v_data, v_target = v_data.to(data.device), v_target.to(data.device)
#                 v_output = model(v_data)
#                 v_loss = nn.CrossEntropyLoss()(v_output, v_target)
#                 total_error += v_loss.item()
#                 count += 1
#             avg_verification_error = total_error / count

#         # Check consistency with neighborhood threshold (rho)
#         if len(self.consistent_batches) == 0 or batch_loss <= avg_verification_error + self.rho:
#             self.consistent_batches.append((data, target))

#     def step(self, model, loss_fn):
#         if self.consistent_batches:
#             model.train()  # Ensure model is in training mode
#             for data, target in self.consistent_batches:
#                 model.zero_grad()
#                 output = model(data)
#                 loss = loss_fn(output, target)
#                 loss.backward()
#                 for group in self.param_groups:
#                     for p in group['params']:
#                         if p.grad is not None:
#                             p.data.add_(p.grad, alpha=-group['lr'])
#             # Clear consistent batches after weight update
#             self.consistent_batches.clear()
