import nemo_run as run

with run.lazy_imports():
    from nemo_run.test_utils import dummy_entrypoint, dummy_recipe


if __name__ == "__main__":
    run.cli.main(dummy_entrypoint, default_factory=dummy_recipe)
