"""A tiny feedforward neural network built from scratch with NumPy.

Run it with:
    python my_playground/ai-learning/nn_toy.py

This teaches the core ideas without hiding them behind a deep-learning library:
1. A layer computes: z = input @ weights + bias
2. An activation adds non-linearity, so the network can learn curved patterns
3. A loss measures how wrong the prediction is
4. Backpropagation calculates how each parameter should change
5. Gradient descent nudges the parameters to reduce the loss
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Squash numbers into the range 0..1."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(sigmoid_output: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid when you already have sigmoid(x)."""
    return sigmoid_output * (1.0 - sigmoid_output)


@dataclass
class ForwardPass:
    """Values saved from the forward pass so backprop can reuse them."""

    x: np.ndarray
    hidden_z: np.ndarray
    hidden_a: np.ndarray
    output_z: np.ndarray
    prediction: np.ndarray


class FeedForwardNeuralNetwork:
    """A 2-layer neural network: inputs -> hidden layer -> output.

    For this toy example:
    - input_size=2 because each sample has two numbers
    - hidden_size=4 gives the network four small pattern detectors
    - output_size=1 because we predict one probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        # Small random weights break symmetry; zeros would make neurons learn
        # the same thing. Biases can safely start at zero.
        self.w1 = rng.normal(loc=0.0, scale=0.5, size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.normal(loc=0.0, scale=0.5, size=(hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def forward(self, x: np.ndarray) -> ForwardPass:
        """Move data left-to-right through the network."""
        hidden_z = x @ self.w1 + self.b1
        hidden_a = sigmoid(hidden_z)

        output_z = hidden_a @ self.w2 + self.b2
        prediction = sigmoid(output_z)

        return ForwardPass(
            x=x,
            hidden_z=hidden_z,
            hidden_a=hidden_a,
            output_z=output_z,
            prediction=prediction,
        )

    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Loss for binary labels, where lower is better."""
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        losses = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(np.mean(losses))

    def backward(self, cache: ForwardPass, y_true: np.ndarray) -> dict[str, np.ndarray]:
        """Calculate gradients for every weight and bias.

        The output layer uses sigmoid + binary cross-entropy, which simplifies
        to: prediction - true_label.
        """
        n_samples = y_true.shape[0]

        d_output_z = (cache.prediction - y_true) / n_samples
        d_w2 = cache.hidden_a.T @ d_output_z
        d_b2 = np.sum(d_output_z, axis=0, keepdims=True)

        d_hidden_a = d_output_z @ self.w2.T
        d_hidden_z = d_hidden_a * sigmoid_derivative(cache.hidden_a)
        d_w1 = cache.x.T @ d_hidden_z
        d_b1 = np.sum(d_hidden_z, axis=0, keepdims=True)

        return {
            "w1": d_w1,
            "b1": d_b1,
            "w2": d_w2,
            "b2": d_b2,
        }

    def update(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
        """Take one gradient descent step."""
        self.w1 -= learning_rate * gradients["w1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.w2 -= learning_rate * gradients["w2"]
        self.b2 -= learning_rate * gradients["b2"]

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 10_000,
        learning_rate: float = 1.0,
        print_every: int = 1_000,
    ) -> None:
        """Repeat forward -> loss -> backward -> update."""
        for epoch in range(1, epochs + 1):
            cache = self.forward(x)
            loss = self.binary_cross_entropy(y, cache.prediction)
            gradients = self.backward(cache, y)
            self.update(gradients, learning_rate)

            if epoch == 1 or epoch % print_every == 0:
                print(f"epoch {epoch:>5} | loss {loss:.4f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return 0/1 labels from probabilities."""
        probabilities = self.forward(x).prediction
        return (probabilities >= 0.5).astype(int)


def make_xor_data() -> tuple[np.ndarray, np.ndarray]:
    """XOR is a classic tiny problem a single straight line cannot solve."""
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array(
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ]
    )
    return x, y


def main() -> None:
    x, y = make_xor_data()
    network = FeedForwardNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    print("Before training:")
    print(network.forward(x).prediction.round(3))

    network.train(x, y, epochs=8_000, learning_rate=1.0, print_every=1_000)

    probabilities = network.forward(x).prediction
    predictions = network.predict(x)

    print("\nAfter training:")
    for inputs, expected, probability, prediction in zip(x, y, probabilities, predictions):
        print(
            f"input={inputs.astype(int).tolist()} "
            f"expected={int(expected[0])} "
            f"probability={probability[0]:.3f} "
            f"prediction={int(prediction[0])}"
        )


if __name__ == "__main__":
    main()
