# Contents of train.py
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
import random

def train(model, device, train_data, validation_data, optimizer, epoch, loss_fn, verification_set_num=4, rho=10, log_interval=10):
    model.train()

    # Step 1: Create a Dummy Verification Set from the Training Data
    total_indices = list(range(len(train_data)))
    verification_indices = random.sample(total_indices, 64 * verification_set_num)  # Dummy set size based on batches
    training_indices = list(set(total_indices) - set(verification_indices))

    dummy_verification_set = Subset(train_data, verification_indices)
    training_set = Subset(train_data, training_indices)

    dummy_verification_loader = DataLoader(dummy_verification_set, batch_size=64, shuffle=True)
    training_loader = DataLoader(training_set, batch_size=64, shuffle=True)

    # Step 2: Iterate through the batches of data within an epoch
    for batch_idx, (data, target) in enumerate(training_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        # Perform weight update for the current data instance
        optimizer.step(model, loss_fn)

        # Step 3: Calculate approximate average error (Ē_t) using dummy verification data
        total_error = 0
        count = 0
        for i, (v_data, v_target) in enumerate(dummy_verification_loader):
            if i >= verification_set_num:
                break  # Limit the number of batches for dummy verification
            v_data, v_target = v_data.to(device), v_target.to(device)
            with torch.no_grad():
                v_output = model(v_data)
                v_loss = loss_fn(v_output, v_target)
                total_error += v_loss.item()
                count += 1
        avg_dummy_verification_error = total_error / count if count > 0 else 0  # Dummy verification error

        # Step 4: Collect consistent data points (ψ) for iterations 1 through `ρ-1`
        if (batch_idx + 1) % rho != 0:  # Collect during 1 to `ρ-1` iterations
            optimizer.collectConsistentBatches(loss.item(), data, target, avg_dummy_verification_error)

        # Step 5: On the `ρ`-th iteration, refine weights using consistent data and reset consistent batches
        if (batch_idx + 1) % rho == 0:
            optimizer.refine_with_consistent_data(model, loss_fn, avg_dummy_verification_error)
            # print(f"Refinement performed at iteration {batch_idx + 1}")

        # Log the training progress every `log_interval` batches
        if batch_idx % log_interval == 0:
            print(f'Epoch: {epoch}, Iteration: {batch_idx + 1}, Loss: {loss.item():.6f}')

    print(f"Epoch {epoch} completed.")



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

