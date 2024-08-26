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
def train_model(model: Model, optimizer: Optimizer, epochs: int = 10, batch_size: int = 32):
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


@run.cli.factory
@run.autoconvert
def my_model(hidden_size: int = 256, num_layers: int = 3, activation: str = "relu") -> Model:
    """
    Create a model configuration.
    """
    return Model(hidden_size=hidden_size, num_layers=num_layers, activation=activation)


@run.cli.factory
@run.autoconvert
def my_optimizer(
    learning_rate: float = 0.001, weight_decay: float = 1e-5, betas: List[float] = [0.9, 0.999]
) -> Optimizer:
    """
    Create an optimizer configuration.
    """
    return Optimizer(learning_rate=learning_rate, weight_decay=weight_decay, betas=betas)


@run.cli.factory
@run.autoconvert
def local_executor() -> run.LocalExecutor:
    return run.LocalExecutor()


@run.cli.entrypoint(type="experiment")
def train_models_experiment(
    ctx: run.cli.RunContext,
    models: List[Model] = [my_model(), my_model(hidden_size=512)],
    optimizers: List[Optimizer] = [my_optimizer(), my_optimizer(learning_rate=0.01)],
    epochs: int = 10,
    batch_size: int = 32,
    sequential: bool = False,
):
    """
    Run an experiment to train multiple models with different configurations.

    Args:
        ctx (run.RunContext): The run context for the experiment.
        models (List[Model]): List of model configurations to train.
        optimizers (List[Optimizer]): List of optimizer configurations to use.
        epochs (int): Number of training epochs for each model.
        batch_size (int): Batch size for training.
    """

    with run.Experiment("train_models_experiment") as exp:
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            train = run.Partial(
                train_model, model=model, optimizer=optimizer, epochs=epochs, batch_size=batch_size
            )

            exp.add(train, name=f"train_model_{i}", executor=ctx.executor)

        ctx.launch(exp, sequential=sequential)


if __name__ == "__main__":
    run.cli.main(train_models_experiment)
