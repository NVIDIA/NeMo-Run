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


@run.cli.entrypoint
def train_model(
    model: Model,
    optimizer: Optimizer,
    epochs: int = 10,
    batch_size: int = 32
):
    """
    Train a model using the specified configuration.

    Args:
        model (Model): Configuration for the model.
        optimizer (Optimizer): Configuration for the optimizer.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        batch_size (int, optional): Batch size for training. Defaults to 32.
    """
    print(f"Training model with the following configuration:")
    print(f"Model: {model}")
    print(f"Optimizer: {optimizer}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    # Simulating model training
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

    print("Training completed!")


@run.cli.factory
@run.autoconvert
def my_model(
    hidden_size: int = 256,
    num_layers: int = 3,
    activation: str = 'relu'
) -> Model:
    """
    Create a model configuration.
    """
    return Model(hidden_size=hidden_size, num_layers=num_layers, activation=activation)


@run.cli.factory
@run.autoconvert
def my_optimizer(
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    betas: List[float] = [0.9, 0.999]
) -> Optimizer:
    """
    Create an optimizer configuration.
    """
    return Optimizer(learning_rate=learning_rate, weight_decay=weight_decay, betas=betas)


@run.cli.entrypoint(type="sequential_experiment")
def train_and_evaluate(
    experiment: run.Experiment,
    executor: run.Executor,
    model: Model = my_model(),
    optimizer: Optimizer = my_optimizer(),
    train_epochs: int = 10,
    eval_epochs: int = 2
):
    """
    Run a sequential experiment to train and evaluate a model.

    Args:
        experiment (run.Experiment): The experiment object.
        executor (run.Executor): The executor for running tasks.
        model (run.Config["Model"]): Configuration for the model.
        optimizer (run.Config["Optimizer"]): Configuration for the optimizer.
        train_epochs (int, optional): Number of training epochs. Defaults to 10.
        eval_epochs (int, optional): Number of evaluation epochs. Defaults to 2.
    """
    train = run.Partial(train_model, model=model, optimizer=optimizer, epochs=train_epochs)
    evaluate = run.Partial(train_model, model=model, optimizer=optimizer, epochs=eval_epochs)

    experiment.add(train, executor=executor, name="train")
    experiment.add(evaluate, executor=executor, name="evaluate")

    return experiment


if __name__ == "__main__":
    run.cli.main(train_and_evaluate)
