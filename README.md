<img src="images/matly-logo-2k.png" alt="Matly" width="200"/>

# MatlyPy

**MatlyPy** is a lightweight Python library that combines the power of [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/) with built-in ML utilities. It provides tools for tensor mathematics, data visualization, and building simple N-gram language models — all under a clean, unified API.

> **Note:** Earlier working names like "pyplot" or "matpy" are taken. This library is **MatlyPy**.

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
git clone https://github.com/fruzino/MatlyPY.git
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
| `softmax(tensor)` | Numerically stable softmax activation |

---

### `plot` — Plotting

The `plot` class wraps Matplotlib for quick data visualization.

| Method | Description |
|---|---|
| `plot(array, height, width, ...)` | Plot the mean of an array along a given axis with explicit size control |
| `autoplot(array, ...)` | Auto-detects array dimensions and plots with sensible defaults |
| `imgplot(data, ...)` | Display image data with optional colormap and colorbar |
| `convert(source, target)` | Convert between file formats (e.g. `.npy` ↔ image) |
| `theme(ax, theme, ...)` | Apply a visual theme (`"dark"` or `"default"`) to an axes object |
| `annotate(ax, text, xy, xytext)` | Add an annotated callout arrow to a plot |
| `fill(ax, x, y1, y2, ...)` | Fill the area between two curves |
| `turtlot.turtplot(data, speed)` | Plot a NumPy (N, 2) array as turtle line movements |
| `turtlot.turtplot_img(path, scale)` | Render an image file pixel-by-pixel using turtle graphics |

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
| `tools.fetch(url)` | Scrape clean text from a URL for use as training data |
| `tools.imagegen(prompt, ...)` | Bridge to a local Stable Diffusion API for image generation |

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

Calculates the angle between two tensors by flattening them and computing the cosine similarity. Both tensors must have the same shape.

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

If the standard deviation is `0`, returns mean-centered data without division.

---

### `math.relu(tensor)`

Applies the ReLU activation function element-wise.

```python
out = math.relu([-3, -1, 0, 2, 5])
# out: [0, 0, 0, 2, 5]
```

---

### `math.softmax(tensor)`

Applies numerically stable Softmax to a tensor, returning a probability distribution. Uses the max-subtraction trick to prevent overflow.

```python
probs = math.softmax([1.0, 2.0, 3.0])
# probs: [0.090, 0.245, 0.665]
```

Returns `None` on error, and returns the input unchanged if it is empty.

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

Auto-detects array shape and plots using a 10×6 figure. For multi-dimensional arrays, plots the mean along axis 0. Ideal for quick inspection.

```python
plot.autoplot(loss_history, title="Loss Over Epochs")
```

---

### `plot.imgplot(data, size, title, labels, grid, cmap, **kwargs)`

Displays an image or 2D array. Automatically adds a colorbar when a `cmap` is specified. Extra keyword arguments are forwarded to `plt.imshow()`.

```python
plot.imgplot(image_array, size=(8, 6), title="Heatmap", cmap="viridis")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | array-like | — | Image or 2D data to display |
| `size` | tuple | `(10, 5)` | Figure size as `(width, height)` |
| `title` | str | `"Untitled"` | Plot title |
| `labels` | dict | `{"x": "X", "y": "Y"}` | Axis labels |
| `grid` | bool | `False` | Show grid |
| `cmap` | str or None | `None` | Matplotlib colormap; also enables colorbar |

---

### `plot.convert(source, target)`

Converts a `.npy` array file into an image file (e.g. PNG, JPG). Values in `[0, 1]` are automatically scaled to `[0, 255]`.

```python
plot.convert("matrix.npy", "output.png")
```

---

### `plot.theme(ax, theme, title, grid, hide_spines)`

Applies a visual theme to an existing Matplotlib axes object.

```python
fig, ax = plt.subplots()
plot.theme(ax, theme="dark", title="My Chart", grid=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ax` | Axes | — | Matplotlib axes to style |
| `theme` | str | `"dark"` | `"dark"` or any other value for default |
| `title` | str or None | `None` | Optional bold title |
| `grid` | bool | `True` | Show subtle dashed grid |
| `hide_spines` | bool | `True` | Hide top and right spines |

---

### `plot.annotate(ax, text, xy, xytext)`

Adds an arrow callout annotation to a data point.

```python
plot.annotate(ax, "Peak", xy=(5, 9.5), xytext=(6, 8))
```

---

### `plot.fill(ax, x, y1, y2, color, alpha)`

Fills the area between two curves (or a curve and a baseline).

```python
plot.fill(ax, x=range(10), y1=upper, y2=lower, color="cyan", alpha=0.2)
```

---

### `plot.turtlot.turtplot(data, speed=3)`

Maps a NumPy array of shape `(N, 2)` to turtle movements, drawing a connected path.

```python
import numpy as np
points = np.array([[0,0],[50,100],[100,0]])
plot.turtlot.turtplot(points, speed=5)
```

> Not intended for commercial use.

---

### `plot.turtlot.turtplot_img(path, scale=1)`

Renders an image file pixel-by-pixel using turtle graphics. Use `scale` to reduce image size for faster rendering.

```python
plot.turtlot.turtplot_img("photo.png", scale=0.2)
```

> Not intended for commercial use.

---

### `model.predict(input_vec, weights, labels)`

Performs a forward pass: multiplies inputs by weights, applies softmax, and returns the top label and its probability.

> **Note:** Internally calls `matmath.tensor_multiply`. Ensure your `mathematics.py` exports this name, or update the call to `tensmultiply`.

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

Update rule: `W = W - rate * outer(inputs, error)` where `error = output - target`.

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
| `type` | int | `1` | `1` = word-level, any other value = character-level |
| `instruct` | str | `""` | Seed/prompt text to continue from |
| `dataset` | str | `None` | Path to a `.txt` file for training data |
| `token` | int | `5` | Number of tokens to generate |
| `n` | int | `3` | N-gram order (2 = bigram, 3 = trigram, etc.) |
| `data` | str | `""` | Inline training text (used when `dataset` is `None`) |
| `temp` | float | `1.0` | Sampling temperature. Lower = more deterministic. `≤ 0.1` = greedy |

**Returns:** `(generated_text: str, brain: dict, vocab: list)`

---

### `model.save(filepath, weights, vocab)`

Serializes weights and vocabulary to a custom `.gguf`-format binary file.

```python
model.save("my_model.gguf", weights, vocab)
```

---

### `model.load(filepath)`

Loads weights and vocabulary from a `.gguf` file saved by `model.save()`.

```python
weights, vocab = model.load("my_model.gguf")
```

**Returns:** `(weights: np.ndarray, vocab: list)` or `(None, None)` on failure.

---

### `model.tools.fetch(url)`

Scrapes and cleans text content from a URL, stripping scripts, styles, and excess whitespace. Useful for building a training corpus from web pages.

```python
text = model.tools.fetch("https://example.com/article")
```

**Returns:** A cleaned plain-text string, or an error message string on failure.

---

### `model.tools.imagegen(prompt, negative_prompt, api_url, steps, cfg_scale, width, height)`

Sends a text prompt to a locally running Stable Diffusion API (e.g. Automatic1111 with `--api` enabled), decodes the result, and saves the image to disk.

```python
filename = model.tools.imagegen(
    prompt="a red fox in a snowy forest",
    api_url="http://127.0.0.1:7860",
    steps=20
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | str | — | Image description |
| `negative_prompt` | str | `""` | Things to exclude from the image |
| `api_url` | str | `"http://127.0.0.1:7860"` | Base URL of the local SD API |
| `steps` | int | `20` | Diffusion steps |
| `cfg_scale` | float | `7.0` | Classifier-free guidance scale |
| `width` | int | `512` | Output image width in pixels |
| `height` | int | `512` | Output image height in pixels |

**Returns:** The saved filename on success, or an error message string on failure.

> Requires a local Stable Diffusion server (e.g. Automatic1111) running with `--api` enabled.

---

## Examples

### Tensor Math

```python
from matlypy import math

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
from matlypy import plot
import numpy as np
import matplotlib.pyplot as plt

# Quick line plot
loss = np.random.exponential(scale=0.5, size=(100, 1)) * np.linspace(1, 0.1, 100).reshape(-1, 1)
plot.autoplot(loss, title="Training Loss", ylabel="Loss", xlabel="Epoch", grid=True)

# Themed plot with annotation and fill
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
plot.theme(ax, theme="dark", title="Sine Wave")
plot.fill(ax, x, y, color="cyan", alpha=0.15)
plot.annotate(ax, "Peak", xy=(1.57, 1.0), xytext=(3, 0.8))
plt.show()
```

### N-Gram Language Model

```python
from matlypy import model

corpus = """
the quick brown fox jumps over the lazy dog
the dog barked at the fox and the fox ran away
"""

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

### Web Scraping for Training Data

```python
from matlypy import model

text = model.tools.fetch("https://en.wikipedia.org/wiki/Natural_language_processing")
output, brain, vocab = model.model(data=text, instruct="language", token=20, n=3)
print(output)
```

### Save and Load a Model

```python
import numpy as np
from matlypy import model

vocab = ["hello", "world", "foo", "bar"]
weights = np.random.rand(len(vocab), len(vocab)).astype(np.float32)

model.save("model.gguf", weights, vocab)

loaded_weights, loaded_vocab = model.load("model.gguf")
print(loaded_vocab)
```

---

## Known Issues

- `model.predict()` internally calls `matmath.tensor_multiply`, but the method in `mathematics.py` is named `tensmultiply`. This will raise an `AttributeError` at runtime. Rename the call in `model.py` to `matmath.tensmultiply` to fix it.
- `plot.convert()` currently only supports `.npy` → image conversion. The reverse (image → `.npy`) is not yet implemented despite the docstring suggesting otherwise.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations, math, linear algebra |
| `matplotlib` | Plotting and visualization |
| `requests` | HTTP requests for `tools.fetch` and `tools.imagegen` |
| `beautifulsoup4` | HTML parsing for `tools.fetch` |
| `Pillow` | Image loading for `turtlot.turtplot_img` |
| `opencv-python` | Image writing for `plot.convert` |
| `turtle` | Standard library turtle graphics |

Install all third-party dependencies:

```bash
pip install numpy matplotlib requests beautifulsoup4 Pillow opencv-python
```

---

## Author

**Akshay Singh** — [limesuggestbox360@gmail.com](mailto:limesuggestbox360@gmail.com)

---

> MatlyPy is experimental and not yet recommended for commercial use.