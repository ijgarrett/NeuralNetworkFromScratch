# Neural Network from Scratch

This project implements a neural network from scratch using NumPy to classify points from a spiral dataset into one of three classes. I completed this project in August 2025, building a full pipeline including data generation, model construction, training loop, backpropagation, and the Adam optimizer.

## Project Overview

The goal is to predict the class of a 2D data point (x, y) based on its position in a spiral pattern. The model is a fully connected neural network trained on this synthetic dataset and is capable of correctly classifying the nonlinear decision boundaries between the three classes.

## Dataset

- Generated using `spiral_data(samples=100, classes=3)`
- 300 total points (100 per class)
- Each point has 2 input features and belongs to one of 3 classes
- Designed to test a network's ability to learn nonlinear boundaries

## Tools and Libraries

- Python
- NumPy (matrix math, activation functions, backpropagation)
- Matplotlib (optional visualization)

## Process and Methodology

### 1. Data Generation
- Used a custom `spiral_data` function to create the dataset
- Each class forms a distinct spiral arm

### 2. Model Architecture
- `Layer_Dense(2, 64)` followed by ReLU
- `Layer_Dense(64, 3)` followed by Softmax
- Outputs a probability distribution over 3 classes

### 3. Forward Pass
- ReLU introduces non-linearity after the first dense layer
- Softmax converts final outputs to class probabilities
- Combined with categorical cross-entropy for stable loss

### 4. Loss
- `Activation_Softmax_Loss_CategoricalCrossEntropy` combines activation and loss
- Measures how far the predicted distribution is from the true label
- Lower loss = better fit to training data

### 5. Backward Pass
- Gradients are manually calculated layer by layer
- Uses chain rule for backpropagation
- Includes gradient descent update logic inside the optimizer

### 6. Optimization
- Used Adam optimizer with:
  - `learning_rate = 0.02`
  - `decay = 1e-5`
  - `beta_1 = 0.9`, `beta_2 = 0.999`
- Includes momentum and RMSProp concepts
- Applies bias correction to account for zero-initialized moments

### 7. Training
- Trained in a loop over epochs
- Forward pass → loss calculation → accuracy → backward pass → update
- Printed progress every 100 epochs

## Final Model Performance

- Accuracy reached ~93% on the training set
- Decision boundary visually fits the spiral shape well
- Fast convergence due to Adam and proper weight initialization

## Files in This Project

- `NeuralNetwork.ipynb`: main notebook with all components:
  - `Layer_Dense`
  - `Activation_ReLU`
  - `Activation_Softmax_Loss_CategoricalCrossEntropy`
  - `Optimizer_Adam`
  - Training loop and results

## Timeline
8/4/25 - 8/6/25

## Future Improvements

- Implement mini-batch training
- Add support for more activation and loss functions
- Extend to deeper networks with more hidden layers
- Train on real-world datasets like MNIST or Fashion-MNIST
- Add support for saving and loading models
