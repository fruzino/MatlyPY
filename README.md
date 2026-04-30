# MatlyPy

**MatlyPy** is a lightweight Python library that combines the power of [NumPy](https://numpy.org/) and [Matplotlib](https://plotlib.org/) with built-in ML utilities. It provides tools for tensor mathematics, data visualization, and building simple N-gram language models — all under a clean, unified API.

**NOTE: If you find any "pyplot" or "matpy" those are names that have not been used these are taken by someone else so please consider these names as MatlyPy!**

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [math — Mathematics](#math--mathematics)
  - [plot — Plotting](#plot--plotting)
  - [model — Modeling](#model--modeling)
- [Full API Reference](#full-api-reference)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [Author](#author)

---

## Installation

```bash
pip install matlypy
```

Or install from source:

```bash
git clone https://github.com/fruzino/MatlyPy
cd matlypy
pip install .
```

> Requires Python 3.8 or higher.

---

## Quick Start

```python
from matlypy import math, plot, model

# Tensor multiplication
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
result = math.tensmultiply(a, b)
print(result)

# Plot data
import numpy as np
data = np.random.rand(100, 5)
plot.autoplot(data, title="Random Data", xlabel="Steps", ylabel="Value")

# Train a simple N-gram language model
output, brain, vocab = model.model(
    data="the cat sat on the mat the cat wore a hat",
    instruct="the cat",
    token=6,
    n=3
)
print(output)
```

---

## Modules

### `math` — Mathematics

The `math` class provides static methods for tensor operations and common ML activation/preprocessing functions.

| Method | Description |
|---|---|
| `tensmultiply(t1, t2)` | Matrix/tensor multiplication using `np.matmul` |
| `tensangle(t1, t2, degree)` | Angle between two tensors (in degrees or radians) |
| `standardize(data)` | Z-score standardization (zero mean, unit variance) |
| `relu(tensor)` | ReLU activation: `max(0, x)` |
| `softmax(tensor)` | Softmax activation over a 1D tensor |

---

### `plot` — Plotting

The `plot` class wraps Matplotlib for quick data visualization.

| Method | Description |
|---|---|
| `plot(array, height, width, ...)` | Plot the mean of an array along a given axis with full size control |
| `autoplot(array, ...)` | Auto-detects array dimensions and plots with sensible defaults |

---

### `model` — Modeling

The `model` class provides tools for building, training, saving, and loading simple ML models.

| Method | Description |
|---|---|
| `predict(input_vec, weights, labels)` | Predict a label given input vector and weight matrix |
| `gethot(tag, tags)` | Generate a one-hot encoded vector for a label |
| `weights(weights, inputs, target, output, rate)` | Perform a single gradient descent weight update |
| `model(type, instruct, dataset, token, n, data, temp)` | Train and run an N-gram language model |
| `save(filepath, weights, vocab)` | Save weights and vocabulary to a `.gguf` binary file |
| `load(filepath)` | Load weights and vocabulary from a `.gguf` binary file |

---

## Full API Reference

### `math.tensmultiply(tensor1, tensor2)`

Multiplies two tensors using matrix multiplication rules (`np.matmul`). The last dimension of `tensor1` must match the first dimension of `tensor2`.

```python
result = math.tensmultiply([[1, 2], [3, 4]], [[5], [6]])
# result: [[17], [39]]
```

**Raises:** `ValueError` if shapes are incompatible.

---

### `math.tensangle(tensor1, tensor2, degree=True)`

Calculates the angle between two tensors by flattening them and computing the cosine similarity.

```python
angle = math.tensangle([1, 0, 0], [0, 1, 0])
# angle: 90.0
```

- `degree=True` → returns angle in degrees  
- `degree=False` → returns angle in radians  
- Returns `0.0` if either tensor has zero norm.

---

### `math.standardize(data)`

Applies Z-score normalization: subtracts the mean and divides by the standard deviation.

```python
normalized = math.standardize([10, 20, 30, 40, 50])
```

If standard deviation is `0`, returns mean-centered data (no division).

---

### `math.relu(tensor)`

Applies the ReLU (Rectified Linear Unit) activation function element-wise.

```python
out = math.relu([-3, -1, 0, 2, 5])
# out: [0, 0, 0, 2, 5]
```

---

### `math.softmax(tensor)`

Applies the numerically stable Softmax function to a tensor, returning a probability distribution.

```python
probs = math.softmax([1.0, 2.0, 3.0])
# probs: [0.090, 0.245, 0.665]
```

---

### `plot.plot(array, height, width, axis=0, title, xlabel, ylabel, grid)`

Plots the mean of an array along the specified axis with explicit figure dimensions.

```python
plot.plot(data, height=6, width=10, axis=0, title="Training Loss", ylabel="Loss")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `array` | array-like | — | Input data |
| `height` | int/float | — | Figure height in inches |
| `width` | int/float | — | Figure width in inches |
| `axis` | int | `0` | Axis along which to compute mean |
| `title` | str | `"Untitled"` | Plot title |
| `xlabel` | str | `"X"` | X-axis label |
| `ylabel` | str | `"Y"` | Y-axis label |
| `grid` | bool | `False` | Show grid |

---

### `plot.autoplot(array, title, xlabel, ylabel, grid)`

Auto-detects array shape and plots using a 10×6 figure. Ideal for quick inspection.

```python
plot.autoplot(loss_history, title="Loss Over Epochs")
```

---

### `model.predict(input_vec, weights, labels)`

Performs a forward pass: multiplies inputs by weights, applies softmax, and returns the top label and its probability.

```python
label, confidence = model.predict(input_vec, weights, labels)
```

---

### `model.gethot(tag, tags)`

Returns a one-hot NumPy vector for the given tag within a list of tags.

```python
vec = model.gethot("cat", ["dog", "cat", "bird"])
# vec: [0., 1., 0.]
```

Returns `None` if the tag is not found.

---

### `model.weights(weights, inputs, target, output, rate=0.01)`

Performs a single weight update step using gradient descent.

```python
updated_w = model.weights(weights, inputs, target, output, rate=0.01)
```

The update rule is: `W = W - rate * outer(inputs, error)` where `error = output - target`.

---

### `model.model(type=1, instruct="", dataset=None, token=5, n=3, data="", temp=1.0)`

Trains and runs an N-gram language model on the provided data.

```python
output, brain, vocab = model.model(
    data="the cat sat on the mat",
    instruct="the",
    token=5,
    n=2,
    temp=0.8
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | int | `1` | `1` = word-level, any other = character-level |
| `instruct` | str | `""` | Prompt/seed text to continue from |
| `dataset` | str | `None` | Path to a `.txt` file to use as training data |
| `token` | int | `5` | Number of tokens to generate |
| `n` | int | `3` | N-gram order (2 = bigram, 3 = trigram, etc.) |
| `data` | str | `""` | Inline training text (used if `dataset` is `None`) |
| `temp` | float | `1.0` | Sampling temperature. Lower = more deterministic |

**Returns:** `(generated_text, brain_dict, vocab_list)`

---

### `model.save(filepath, weights, vocab)`

Serializes weights and vocabulary to a binary `.gguf` file.

```python
model.save("my_model.gguf", weights, vocab)
```

---

### `model.load(filepath)`

Loads weights and vocabulary from a `.gguf` file.

```python
weights, vocab = model.load("my_model.gguf")
```

**Returns:** `(weights: np.ndarray, vocab: list)` or `(None, None)` on failure.

---

## Examples

### Tensor Math

```python
from pyplot import math

# Matrix multiply
A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8], [9, 10], [11, 12]]
print(math.tensmultiply(A, B))

# Angle between vectors
print(math.tensangle([1, 0], [0, 1]))  # 90.0 degrees

# Standardize a dataset
import numpy as np
data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print(math.standardize(data))

# Activation functions
print(math.relu([-2, -1, 0, 1, 2]))
print(math.softmax([1.0, 2.0, 3.0]))
```

### Visualization

```python
from pyplot import plot
import numpy as np

# Simulate training loss
loss = np.random.exponential(scale=0.5, size=(100, 1)) * np.linspace(1, 0.1, 100).reshape(-1, 1)
plot.autoplot(loss, title="Training Loss", ylabel="Loss", xlabel="Epoch", grid=True)
```

### N-Gram Language Model

```python
from pyplot import model

corpus = """
the quick brown fox jumps over the lazy dog
the dog barked at the fox and the fox ran away
"""

# Train a trigram model and generate text
output, brain, vocab = model.model(
    data=corpus,
    instruct="the quick",
    token=8,
    n=3,
    temp=0.7
)
print("Generated:", output)
print("Vocab size:", len(vocab))
```

### Save and Load a Model

```python
import numpy as np
from pyplot import model

# Create dummy weights
vocab = ["hello", "world", "foo", "bar"]
weights = np.random.rand(len(vocab), len(vocab)).astype(np.float32)

model.save("model.gguf", weights, vocab)

loaded_weights, loaded_vocab = model.load("model.gguf")
print(loaded_vocab)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations, math, linear algebra |
| `plotlib` | Plotting and visualization |

Install all dependencies:

```bash
pip install numpy plotlib
```

---

## Author

**Akshay Singh** — [limesuggestbox360@gmail.com](mailto:limesuggestbox360@gmail.com)

---

> pyplot is open and minimal by design. It's meant to be readable, hackable, and a solid foundation for learning ML from scratch.
