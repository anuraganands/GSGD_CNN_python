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
import torch
import torch.optim as optim

class GSGDOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, rho=10, method='sgd', momentum=0.9, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Args:
            lr (float): Learning rate.
            rho (int): Neighborhood allocation parameter.
            method (str): The gradient descent method to use: 'sgd', 'momentum', or 'adam'.
            momentum (float): Momentum factor (for momentum-based SGD).
            beta1 (float): Coefficient for the first moment estimate in Adam.
            beta2 (float): Coefficient for the second moment estimate in Adam.
            eps (float): Small epsilon value to avoid division by zero in Adam.
        """
        defaults = dict(lr=lr, rho=rho, method=method, momentum=momentum, beta1=beta1, beta2=beta2, eps=eps)
        super(GSGDOptimizer, self).__init__(params, defaults)
        self.consistent_batches = []  # Store consistent data instances

    def collectConsistentBatches(self, batch_loss, data, target, avg_dummy_verification_error):
        # Collect consistent data points based on loss comparison
        if batch_loss <= avg_dummy_verification_error:
            self.consistent_batches.append((data, target))

    def step(self, model, loss_fn):
        """Performs a single optimization step on current gradients."""
        method = self.param_groups[0]['method']
        if method == 'momentum':
            self._momentum_step()
        elif method == 'adam':
            self._adam_step()
        else:
            self._sgd_step()

    def _sgd_step(self):
        """Standard SGD update without momentum or adaptive learning."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-group['lr'])

    def _momentum_step(self):
        """SGD with momentum."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Initialize velocity for each parameter
                    if 'velocity' not in self.state[p]:
                        self.state[p]['velocity'] = torch.zeros_like(p.data)
                    
                    velocity = self.state[p]['velocity']
                    momentum = group['momentum']
                    velocity.mul_(momentum).add_(p.grad)  # Update velocity
                    p.data.add_(velocity, alpha=-group['lr'])  # Update weights

    def _adam_step(self):
        """Adam optimization update."""
        for group in self.param_groups:
            beta1, beta2, eps = group['beta1'], group['beta2'], group['eps']
            lr = group['lr']
            for p in group['params']:
                if p.grad is not None:
                    # Initialize state for each parameter
                    if 'm' not in self.state[p]:
                        self.state[p]['m'] = torch.zeros_like(p.data)
                        self.state[p]['v'] = torch.zeros_like(p.data)
                    
                    m, v = self.state[p]['m'], self.state[p]['v']
                    m.mul_(beta1).add_(p.grad, alpha=1 - beta1)  # Update biased first moment estimate
                    v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)  # Update biased second moment estimate
                    m_hat = m / (1 - beta1)  # Correct bias for first moment
                    v_hat = v / (1 - beta2)  # Correct bias for second moment
                    p.data.addcdiv_(m_hat, (v_hat.sqrt() + eps), value=-lr)  # Update weights

    def refine_with_consistent_data(self, model, loss_fn, avg_dummy_verification_error):
        """Refine weights using the consistent data points collected in `ψ`."""
        if self.consistent_batches:
            model.train()  # Ensure model is in training mode
            for data, target in self.consistent_batches:
                model.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                
                # Apply chosen update method to consistent batches
                self.step(model, loss_fn)
            # Clear consistent batches after refinement
            self.consistent_batches.clear()





# class GSGDOptimizer(optim.Optimizer):
#     def __init__(self, params, lr=0.01, revisit_batch_num=5):
#         defaults = dict(lr=lr)
#         super(GSGDOptimizer, self).__init__(params, defaults)
#         self.consistent_batches = []  # Store consistent data instances

#     def collect_consistent_batches(self, batch_loss, data, target, avg_dummy_verification_error):
#         # Check if the batch_loss is within acceptable bounds to be considered consistent
#         if batch_loss <= avg_dummy_verification_error:
#             self.consistent_batches.append((data, target))

#     def step(self, model, loss_fn):
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is not None:
#                     p.data.add_(p.grad, alpha=-group['lr'])

#     def refine_with_consistent_data(self, model, loss_fn):
#         if self.consistent_batches:
#             model.train()  # Ensure model is in training mode
#             for data, target in self.consistent_batches:
#                 model.zero_grad()
#                 output = model(data)
#                 loss = loss_fn(output, target)
#                 loss.backward()
                
#                 # Apply gradients to parameters
#                 for group in self.param_groups:
#                     for p in group['params']:
#                         if p.grad is not None:
#                             p.data.add_(p.grad, alpha=-group['lr'])
#             # Clear consistent batches after weight update
#             self.consistent_batches.clear()

