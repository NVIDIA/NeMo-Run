from dataclasses import dataclass
from typing import List

import nemo_run as run


@dataclass
class Model:
    """Dummy model config"""

    hidden_size: int
    num_layers: int
    activation: str


@dataclass
class Optimizer:
    """Dummy optimizer config"""

    learning_rate: float
    weight_decay: float
    betas: List[float]


@run.cli.factory
@run.autoconvert
def my_model(hidden_size: int = 256, num_layers: int = 3, activation: str = "relu") -> Model:
    """
    Create a model configuration.
    """
    return Model(hidden_size=hidden_size, num_layers=num_layers, activation=activation)


@run.cli.factory
def my_optimizer(
    learning_rate: float = 0.001, weight_decay: float = 1e-5, betas: List[float] = [0.9, 0.999]
) -> run.Config[Optimizer]:
    """Create an optimizer configuration."""
    return run.Config(
        Optimizer, learning_rate=learning_rate, weight_decay=weight_decay, betas=betas
    )


def train_model(
    model: Model,
    optimizer: Optimizer,
    epochs: int = 10,
    batch_size: int = 32,
):
    """
    Train a model using the specified configuration.

    Args:
        model (Model): Configuration for the model.
        optimizer (Optimizer): Configuration for the optimizer.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        batch_size (int, optional): Batch size for training. Defaults to 32.
    """
    print("Training model with the following configuration:")
    print(f"Model: {model}")
    print(f"Optimizer: {optimizer}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    # Simulating model training
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

    print("Training completed!")


def custom_defaults() -> run.Partial[train_model]:
    return run.Partial(
        train_model,
        model=my_model(hidden_size=512),
        optimizer=my_optimizer(learning_rate=0.0005),
        epochs=50,
        batch_size=2048,
    )


@run.autoconvert
def local_executor() -> run.Executor:
    return run.LocalExecutor()


class DummyPlugin(run.Plugin):
    def setup(self, task: run.Partial[train_model], executor: run.Executor):
        task.epochs *= 2


if __name__ == "__main__":
    run.cli.main(
        train_model,
        default_factory=custom_defaults,
        default_executor=local_executor(),
        default_plugins=run.Config(DummyPlugin),
    )
