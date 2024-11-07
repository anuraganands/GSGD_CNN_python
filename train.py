# Contents of train.py
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
import random

def train(model, device, train_data, validation_data, optimizer, epoch, loss_fn, verification_set_num=4):
    model.train()

    # Step 1: Create a Dummy Verification Set from the Training Data
    total_indices = list(range(len(train_data)))
    verification_indices = random.sample(total_indices, 64 * verification_set_num)  # Dummy set size based on batches
    training_indices = list(set(total_indices) - set(verification_indices))

    dummy_verification_set = Subset(train_data, verification_indices)
    training_set = Subset(train_data, training_indices)

    dummy_verification_loader = DataLoader(dummy_verification_set, batch_size=64, shuffle=True)
    training_loader = DataLoader(training_set, batch_size=64, shuffle=True)

    # Step 2: Calculate the Dummy Verification Error for Consistency Checking
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

    # Step 3: Training with Consistent Batch Selection Based on Dummy Verification Error
    for batch_idx, (data, target) in enumerate(training_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        # Collect consistent batches based on dummy verification error
        optimizer.collect_consistent_batches(loss.item(), data, target, avg_dummy_verification_error)

        # Update weights with consistent batches
        optimizer.step(model, loss_fn)

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(training_loader.dataset)}] '
                  f'Loss: {loss.item():.6f}')

    # Step 4: Full Validation Evaluation (for guiding training)
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in DataLoader(validation_data, batch_size=64, shuffle=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += loss_fn(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(DataLoader(validation_data).dataset)
    print(f'\nValidation set: Average loss: {validation_loss:.4f}, '
          f'Accuracy: {correct}/{len(DataLoader(validation_data).dataset)} '
          f'({100. * correct / len(DataLoader(validation_data).dataset):.0f}%)\n')


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

