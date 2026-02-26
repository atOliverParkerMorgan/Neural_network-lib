# Neural Network Library

## Project Overview
This project is a lightweight neural network library written in pure Python. It includes a custom matrix implementation and a feed-forward neural network with sigmoid activation, backpropagation training (for a single hidden layer), and simple genetic operations for evolving model weights.

The code is designed for learning, experimentation, and small projects where understanding the internals of neural networks is more important than framework-level performance.

## Features
- Custom `Matrix` class with core operations: add, subtract, multiply, divide, transpose, map, and conversion to/from arrays.
- `NeuralNetwork` class supporting:
    - Multiple hidden layers for inference (`feed_forward`).
    - Backpropagation training via `train` (currently limited to one hidden layer).
    - Model copy and weight mutation for evolutionary experiments.
- Basic genetic utilities:
    - Population initialization.
    - Crossover of network parameters.
- Included XOR training example in comments inside `nn.py`.

## Architecture / Structure
```
Neural_network-lib/
├── matrix.py      # Matrix operations used by the neural network
├── nn.py          # NeuralNetwork implementation + training/genetic logic
├── README.md
└── LICENSE.md
```

Core design:
- `Matrix` is the low-level math engine.
- `NeuralNetwork` composes matrices for weights/biases and implements forward pass, training loop, and evolutionary helpers.

## Build & Run Instructions
### Prerequisites
- Python 3.8+

### Setup
No third-party dependencies are required.

```bash
git clone https://github.com/<your-username>/Neural_network-lib.git
cd Neural_network-lib
```

### Minimal usage example
```python
from nn import NeuralNetwork

# XOR setup
nn = NeuralNetwork(2, [2], 1)
inputs = [[1, 0], [0, 1], [1, 1], [0, 0]]
targets = [[1], [1], [0], [0]]

nn.train(inputs, targets, learning_rate=0.01, epoch=500000, ran=True)
print(nn.feed_forward([1, 0]))
```

Note: `train` currently supports only networks with one hidden layer.

## Testing
This repository does not currently include an automated test suite.

Recommended manual checks:
- Run the XOR example and verify outputs are close to expected values.
- Validate matrix dimension behavior using small known inputs.
- Confirm `mutate` and `cross_over` produce valid network instances.

If you plan to extend the project, adding `pytest`-based unit tests for `matrix.py` and basic network behavior is a good next step.

## Project Context
This library was created as a hands-on learning project inspired by neural network educational material from The Coding Train.

It has also been used in a related project:
- [Snake_nn](https://github.com/Plotun333/Snake_nn)
