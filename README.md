# Guided Stochastic Gradient Descent (GSGD) for Convolutional Neural Networks (CNN)
# *NOT YET VERIFIED*

This project implements a Convolutional Neural Network (CNN) with a Guided Stochastic Gradient Descent (GSGD) optimizer in Python. This Python version is adapted from an original MATLAB implementation, focusing on improving classification accuracy and convergence in CNNs by strategically guiding SGD to prioritize consistent training batches.

## Project Structure
- **model.py**: Defines the CNN architecture and the GSGD optimizer class.
- **train.py**: Contains the `train` and `test` functions for training and evaluating the model.
- **main.ipynb**: Main script for data loading, model initialization, and execution.

## Requirements
This implementation uses PyTorch. Install the required packages via:
```bash
pip install torch torchvision
```

## Setup and Usage
Download and Prepare Data: This example uses the MNIST dataset, automatically downloaded by the torchvision.datasets module.
Run the Training Script: Start training by running:
```bash
python main.ipynb
```
## File Details
- **model.py**: Contains the CNN_GSGD class, defining the CNN layers, and GSGDOptimizer class, implementing guided stochastic gradient descent.
- **train.py**: The train function handles the training loop, and test function evaluates model performance on the test set.
- **main.ipynb**: Loads the dataset, initializes the model and optimizer, and starts the training loop.

## Parameters and Hyperparameters
- Major ones:
  - *lr*: Learning rate for the optimizer, set in main.ipynb.
  - *rho*: Neighborhood size in GSGDOptimizer for identifying consistent batches.
  - *batch_size*: Batch size for training and testing.
- Minor ones:
  - revisit_batch_num: how many consistent batches to revisit for weight update. Defined in the constructor of GSGDOptimizer.
  - verification_set_num: a small dummy validation set to indicate if a batch is consistent or not. Used for efficiency purpose. Defined in the *train* function in *train.py*
- Feel free to adjust these hyperparameters in main.ipynb for experimentation.

## Results and Evaluation
The code reports training loss and accuracy after each epoch. You can modify main.ipynb to save the model or log additional metrics if needed.

## Example Output
Expected output after training includes training loss and test accuracy printed to the console. Here’s an example of the expected output format:

Train Epoch: 1 [0/60000]  Loss: 0.465
...
Test set: Average loss: 0.0264, Accuracy: 9912/10000 (99%)

## References
This implementation is based on the paper: "A Strategic Weight Refinement Maneuver for Convolutional Neural Networks".



