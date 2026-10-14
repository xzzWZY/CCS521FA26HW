# Problem 4: MNIST Interval Analysis

Train the starter network for 15 epochs and evaluate interval-verified accuracy
on all 10,000 MNIST test images at radii 0.01–0.10.

## Run with uv

From the repository root:

```bash
cd hw2/q4
uv sync --frozen
uv run jupyter lab mnist_interval_analysis.ipynb
```

Select the Python 3 kernel and **Restart Kernel and Run All Cells**.
The first run downloads MNIST. Each full run retrains the model on CPU (seed 42).

To execute and save outputs without opening a browser:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 mnist_interval_analysis.ipynb
```

## Files

- `mnist_interval_analysis.ipynb`: training, interval analysis, and results.
- `results/`: model weights, JSON metrics, CSV results, and accuracy plot.

The verifier bounds logit differences, which preserve softmax class ordering.
A failed certificate does not establish an adversarial example.
