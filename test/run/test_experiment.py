import pytest
from fiddle._src.experimental.serialization import UnserializableValueError

import nemo_run as run
from test.dummy_factory import DummyModel, DummyTrainer, dummy_train


@pytest.fixture
def experiment(tmpdir):
    return run.Experiment("dummy_experiment", base_dir=tmpdir)


class TestValidateTask:
    def test_validate_task(self, experiment: run.Experiment):
        experiment._validate_task("valid_script", run.Script(inline="echo 'hello world'"))

        valid_partial = run.Partial(
            dummy_train, dummy_model=run.Config(DummyModel), dummy_trainer=run.Config(DummyTrainer)
        )
        experiment._validate_task("valid_partial", valid_partial)

        invalid_partial = run.Partial(
            dummy_train, dummy_model=DummyModel(), dummy_trainer=DummyTrainer()
        )
        with pytest.raises(UnserializableValueError):
            experiment._validate_task("invalid_partial", invalid_partial)
