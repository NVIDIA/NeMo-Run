import nemo_run as run

base = run.Experiment.from_id("train_models_experiment_1724757191")

if __name__ == "__main__":
    with base.reset() as experiment:
        experiment.tasks[0].epochs = 15

        experiment.run(sequential=True)
