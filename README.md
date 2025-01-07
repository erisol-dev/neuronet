
# NeuralNetwork Class

This is a Python implementation of a customizable fully connected feed-forward neural network with support for training, evaluation, and gradient-based optimization. It uses NumPy for numerical computations.

## Features
- Support for customizable architecture (number of layers and neurons per layer).
- Implements ReLU activation and its derivative for backpropagation.
- Mini-batch gradient descent for training.
- Training/validation data split for evaluation.
- Performance metrics (accuracy) displayed after each epoch.
- Fully configurable learning rate, batch size, and number of epochs.

---

## Installation

1. Clone or download the repository:
   ```bash
   git clone https://github.com/erisol-dev/neuronet.git
   ```

2. Install required dependencies:
   ```bash
   pip install -r .\requirements.txt
   ```

---

## Usage

### Import the Class
To use the `NeuralNetwork` class, import it into your project:
```python
from neuronet import NeuralNetwork
```

### Initialize the Network
You can define the architecture of the network by specifying the number of neurons in each layer via the `layers` parameter.

```python
import numpy as np

# Example: Initialize the network
input_data = np.random.rand(1000, 784)  # Example input (1000 images, flattened 28x28 pixels)
labels = np.random.randint(0, 10, size=(1000,))  # Example labels for 10 classes

# Define the architecture (e.g., input layer: 784, hidden layers: [128, 64], output layer: 10)
layers = [784, 128, 64, 10]

# Create an instance of the NeuralNetwork
nn = NeuralNetwork(input=input_data, labels=labels, layers=layers)
```

### Train the Network
Train the network using the `train` method. You can specify the batch size, number of epochs, and learning rate.

```python
nn.train(batch_size=32, epochs=20, learning_rate=0.01)
```

### Evaluate the Network
The `evaluate` method is automatically called after each epoch during training. It prints the accuracy on the test set.

---

## Parameters

### Initialization Parameters
- `input` (numpy array): The input data (e.g., images or features).
- `labels` (numpy array): The corresponding labels for the input data.
- `layers` (list): A list specifying the number of neurons in each layer (including input and output layers).
- `normalization` (int): A number that you can use to normalize your input values (for mnist 255.0 for pixel values).
- `ratio` (float, default=0.8): Ratio for splitting the dataset into training and testing sets.
- `random_seed` (int, default=72): Random seed for reproducibility.

### Training Parameters
- `batch_size` (int, default=5): Number of training examples per mini-batch.
- `epochs` (int, default=30): Number of passes over the training dataset.
- `learning_rate` (float, default=0.012): Step size for gradient descent.

---

## Methods

### `__init__(self, input, labels, layers, ratio=0.8, random_seed=72)`
Initializes the neural network with given architecture and parameters.

### `train(self, batch_size=5, epochs=30, learning_rate=0.012)`
Trains the network using mini-batch gradient descent.

### `forward(self, image)`
Performs forward propagation for a given input image.

### `calculate_errors(self, errors)`
Computes the error (delta) for each layer during backpropagation.

### `apply_gradient(self, average_weights, average_bias, learning_rate)`
Applies gradient updates to the weights and biases.

### `randomize_training_data(self)`
Shuffles the training data to improve model generalization.

### `relu(self, x, alpha=0.01)`
Applies the ReLU activation function.

### `derivative_relu(self, x, alpha=0.01)`
Computes the derivative of the ReLU activation function.

### `cost(self, output, image_label)`
Calculates the mean squared error between the predicted and true labels.

### `evaluate(self, _)`
Evaluates the model's performance on the test set and prints accuracy.

---

## Example Usage

```python
# Example Dataset (MNIST-like)
input_data = np.random.rand(1000, 784)  # 1000 samples of 28x28 images flattened
labels = np.random.randint(0, 10, size=(1000,))  # 10 classes

# Initialize Neural Network
layers = [784, 128, 64, 10]  # Input, Hidden1, Hidden2, Output
nn = NeuralNetwork(input=input_data, labels=labels, layers=layers)

# Train the Network
nn.train(batch_size=32, epochs=10, learning_rate=0.01)
```

---

## Notes

1. **Data Normalization**: The input data is normalized to the range `[0, 1]` by dividing by `255.0`.
2. **Dataset Splitting**: The dataset is split into training and testing sets based on the `ratio` parameter.
3. **Reproducibility**: Random seed ensures consistent initialization of weights and biases.
4. **Activation Function**: ReLU is used as the activation function, and its derivative is used for backpropagation.

---

## Limitations
- This implementation only supports the ReLU activation function.
- Gradient-based optimization does not use advanced techniques like momentum, RMSprop, or Adam.
- The cost function is Mean Squared Error (MSE), which may not be optimal for classification tasks compared to Cross-Entropy Loss.

---

## Future Enhancements
- Add support for additional activation functions (e.g., sigmoid, tanh).
- Implement advanced optimizers (e.g., Adam, SGD with momentum).
- Allow for configurable loss functions (e.g., cross-entropy loss).
- Support dropout and regularization techniques.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
