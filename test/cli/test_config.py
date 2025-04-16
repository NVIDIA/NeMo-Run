import pytest
import json
import toml
import yaml
from typing import Any, Dict

from nemo_run.config import Config, Partial
from nemo_run.cli.config import ConfigSerializer

# --- Test Fixtures and Helper Data ---


class DummyClass:
    """A simple class for testing Config serialization."""

    def __init__(self, a: int, b: str = "default", c: Any = None):
        self.a = a
        self.b = b
        self.c = c

    def __eq__(self, other):
        if not isinstance(other, DummyClass):
            return NotImplemented
        return self.a == other.a and self.b == other.b and self.c == other.c


def dummy_func(x: float, y: bool = True) -> float:
    """A simple function for testing Partial serialization."""
    return x if y else -x


@pytest.fixture
def sample_config() -> Config:
    """Provides a sample Config object for testing."""
    nested_partial = Partial(dummy_func, x=1.0)
    return Config(DummyClass, a=10, b="test", c=nested_partial)


@pytest.fixture
def sample_partial() -> Partial:
    """Provides a sample Partial object for testing."""
    return Partial(dummy_func, x=3.14, y=False)


@pytest.fixture
def sample_dict() -> Dict[str, Any]:
    """Provides a sample dictionary for testing."""
    return {
        "section1": {"key1": "value1", "key2": 123, "list": [1, "a", None]},
        "section2": {"nested": {"a": True, "b": None}},
        "top_level": 45.6,
        "another_list": [10, 20],
    }


@pytest.fixture
def serializer() -> ConfigSerializer:
    """Provides a ConfigSerializer instance."""
    return ConfigSerializer()


# --- Test Class for ConfigSerializer ---


class TestConfigSerializer:
    # --- YAML Tests ---
    def test_serialize_deserialize_yaml(self, serializer, sample_config, tmp_path):
        yaml_str = serializer.serialize_yaml(sample_config)
        assert "_target_: test.cli.test_config.DummyClass" in yaml_str  # Updated target key
        assert "a: 10" in yaml_str
        assert "b: test" in yaml_str
        assert "_partial_: true" in yaml_str
        assert "_target_: test.cli.test_config.dummy_func" in yaml_str

        deserialized_config = serializer.deserialize_yaml(yaml_str)
        assert isinstance(deserialized_config, Config)
        assert deserialized_config.a == 10
        assert deserialized_config.b == "test"
        assert isinstance(deserialized_config.c, Partial)
        assert deserialized_config.c.x == 1.0
        assert deserialized_config.c.y

        file_path = tmp_path / "config.yaml"
        serializer.dump_yaml(sample_config, file_path)
        assert file_path.exists()

        loaded_config = serializer.load_yaml(file_path)
        assert isinstance(loaded_config, Config)
        assert loaded_config.a == 10
        assert loaded_config.b == "test"
        assert isinstance(loaded_config.c, Partial)
        assert loaded_config.c.x == 1.0
        assert loaded_config.c.y

    def test_serialize_deserialize_partial_yaml(self, serializer, sample_partial, tmp_path):
        yaml_str = serializer.serialize_yaml(sample_partial)
        assert "_partial_: true" in yaml_str
        assert "_target_: test.cli.test_config.dummy_func" in yaml_str
        assert "x: 3.14" in yaml_str
        assert "y: false" in yaml_str

        deserialized_partial = serializer.deserialize_yaml(yaml_str)
        assert isinstance(deserialized_partial, Partial)
        assert deserialized_partial.x == 3.14
        assert not deserialized_partial.y

        file_path = tmp_path / "partial.yaml"
        serializer.dump_yaml(sample_partial, file_path)
        assert file_path.exists()

        loaded_partial = serializer.load_yaml(file_path)
        assert isinstance(loaded_partial, Partial)
        assert loaded_partial.x == 3.14
        assert not loaded_partial.y

    # --- JSON Tests ---
    def test_serialize_deserialize_json(self, serializer, sample_config, tmp_path):
        json_str = serializer.serialize_json(sample_config)
        json_data = json.loads(json_str)
        assert json_data["_target_"] == "test.cli.test_config.DummyClass"
        assert json_data["a"] == 10
        assert json_data["b"] == "test"
        assert json_data["c"]["_partial_"] is True
        assert json_data["c"]["_target_"] == "test.cli.test_config.dummy_func"

        deserialized_config = serializer.deserialize_json(json_str)
        assert isinstance(deserialized_config, Config)
        assert deserialized_config.a == 10
        assert deserialized_config.b == "test"
        assert isinstance(deserialized_config.c, Partial)
        assert deserialized_config.c.x == 1.0
        assert deserialized_config.c.y

        file_path = tmp_path / "config.json"
        serializer.dump_json(sample_config, file_path)
        assert file_path.exists()

        loaded_config = serializer.load_json(file_path)
        assert isinstance(loaded_config, Config)
        assert loaded_config.a == 10
        assert loaded_config.b == "test"
        assert isinstance(loaded_config.c, Partial)
        assert loaded_config.c.x == 1.0
        assert loaded_config.c.y

    def test_serialize_deserialize_partial_json(self, serializer, sample_partial, tmp_path):
        json_str = serializer.serialize_json(sample_partial)
        json_data = json.loads(json_str)
        assert json_data["_partial_"] is True
        assert json_data["_target_"] == "test.cli.test_config.dummy_func"
        assert json_data["x"] == 3.14
        assert json_data["y"] is False

        deserialized_partial = serializer.deserialize_json(json_str)
        assert isinstance(deserialized_partial, Partial)
        assert deserialized_partial.x == 3.14
        assert not deserialized_partial.y

        file_path = tmp_path / "partial.json"
        serializer.dump_json(sample_partial, file_path)
        assert file_path.exists()

        loaded_partial = serializer.load_json(file_path)
        assert isinstance(loaded_partial, Partial)
        assert loaded_partial.x == 3.14
        assert not loaded_partial.y

    # --- TOML Tests ---
    def test_serialize_deserialize_toml(self, serializer, sample_config, tmp_path):
        toml_str = serializer.serialize_toml(sample_config)
        toml_data = toml.loads(toml_str)
        assert toml_data["_target_"] == "test.cli.test_config.DummyClass"
        assert toml_data["a"] == 10
        assert toml_data["b"] == "test"
        assert toml_data["c"]["_partial_"] is True
        assert toml_data["c"]["_target_"] == "test.cli.test_config.dummy_func"

        deserialized_config = serializer.deserialize_toml(toml_str)
        assert isinstance(deserialized_config, Config)
        assert deserialized_config.a == 10
        assert deserialized_config.b == "test"
        assert isinstance(deserialized_config.c, Partial)
        assert deserialized_config.c.x == 1.0
        assert deserialized_config.c.y

        file_path = tmp_path / "config.toml"
        serializer.dump_toml(sample_config, file_path)
        assert file_path.exists()

        loaded_config = serializer.load_toml(file_path)
        assert isinstance(loaded_config, Config)
        assert loaded_config.a == 10
        assert loaded_config.b == "test"
        assert isinstance(loaded_config.c, Partial)
        assert loaded_config.c.x == 1.0
        assert loaded_config.c.y

    def test_serialize_deserialize_partial_toml(self, serializer, sample_partial, tmp_path):
        toml_str = serializer.serialize_toml(sample_partial)
        toml_data = toml.loads(toml_str)
        assert toml_data["_partial_"] is True
        assert toml_data["_target_"] == "test.cli.test_config.dummy_func"
        assert toml_data["x"] == 3.14
        assert toml_data["y"] is False

        deserialized_partial = serializer.deserialize_toml(toml_str)
        assert isinstance(deserialized_partial, Partial)
        assert deserialized_partial.x == 3.14
        assert not deserialized_partial.y

        file_path = tmp_path / "partial.toml"
        serializer.dump_toml(sample_partial, file_path)
        assert file_path.exists()

        loaded_partial = serializer.load_toml(file_path)
        assert isinstance(loaded_partial, Partial)
        assert loaded_partial.x == 3.14
        assert not loaded_partial.y

    # --- Generic Load/Dump Tests ---
    @pytest.mark.parametrize(
        "ext, loader, dumper, cfg_fixture",
        [
            ("yaml", "load_yaml", "dump_yaml", "sample_config"),
            ("yml", "load_yaml", "dump_yaml", "sample_config"),
            ("json", "load_json", "dump_json", "sample_config"),
            ("toml", "load_toml", "dump_toml", "sample_config"),
            ("yaml", "load_yaml", "dump_yaml", "sample_partial"),
            ("yml", "load_yaml", "dump_yaml", "sample_partial"),
            ("json", "load_json", "dump_json", "sample_partial"),
            ("toml", "load_toml", "dump_toml", "sample_partial"),
        ],
    )
    def test_load_dump_extensions(
        self, request, serializer, tmp_path, ext, loader, dumper, cfg_fixture
    ):
        cfg = request.getfixturevalue(cfg_fixture)  # Get the fixture value by name
        file_path = tmp_path / f"{cfg_fixture}.{ext}"

        # Use generic dump
        serializer.dump(cfg, file_path)
        assert file_path.exists()

        # Read back using specific loader to verify dump format
        specific_loader_func = getattr(serializer, loader)
        loaded_config_specific = specific_loader_func(file_path)
        if cfg_fixture == "sample_config":
            assert isinstance(loaded_config_specific, Config)
            assert loaded_config_specific.a == 10
            assert loaded_config_specific.b == "test"
            assert isinstance(loaded_config_specific.c, Partial)
            assert loaded_config_specific.c.x == 1.0
            assert loaded_config_specific.c.y
        elif cfg_fixture == "sample_partial":
            assert isinstance(loaded_config_specific, Partial)
            assert loaded_config_specific.x == 3.14
            assert not loaded_config_specific.y

        # Use generic load
        loaded_config_generic = serializer.load(file_path)
        assert isinstance(loaded_config_generic, (Config, Partial))
        if cfg_fixture == "sample_config":
            assert loaded_config_generic.a == 10
            assert loaded_config_generic.b == "test"
            assert isinstance(loaded_config_generic.c, Partial)
            assert loaded_config_generic.c.x == 1.0
            assert loaded_config_generic.c.y
        elif cfg_fixture == "sample_partial":
            assert loaded_config_generic.x == 3.14
            assert not loaded_config_generic.y

    def test_load_unsupported_extension(self, serializer, tmp_path):
        file_path = tmp_path / "config.txt"
        file_path.touch()
        with pytest.raises(ValueError, match="Unsupported file extension: .txt"):
            serializer.load(file_path)

    def test_dump_unsupported_extension(self, serializer, sample_config, tmp_path):
        file_path = tmp_path / "config.txt"
        with pytest.raises(ValueError, match="Unsupported file extension: .txt"):
            serializer.dump(sample_config, file_path)

    # --- Dictionary Load/Dump Tests ---
    @pytest.mark.parametrize(
        "ext, loader, dumper",
        [
            ("yaml", yaml.safe_load, yaml.safe_dump),
            ("yml", yaml.safe_load, yaml.safe_dump),
            ("json", json.load, json.dump),
            (
                "toml",
                toml.load,
                lambda d, f: f.write(toml.dumps(d)),
            ),  # Use lambda for toml dump signature
        ],
    )
    def test_load_dump_dict(self, serializer, sample_dict, tmp_path, ext, loader, dumper):
        file_path = tmp_path / f"data.{ext}"

        # Create a TOML-compatible version for the TOML test case
        toml_compatible_dict = {
            "section1": {"key1": "value1", "key2": 123},  # Removed 'list'
            "section2": {"nested": {"a": True}},  # Removed 'b': None
            "top_level": sample_dict["top_level"],
            "another_list": sample_dict["another_list"],
        }

        # Determine which dictionary to use based on the format
        dict_to_test = toml_compatible_dict if ext == "toml" else sample_dict

        # Use generic dump_dict with the appropriate dictionary
        serializer.dump_dict(dict_to_test, file_path)
        assert file_path.exists()

        # Verify content using native loaders - still skip for TOML
        # as the internal YAML conversion might slightly change representation
        # compared to direct toml.dump. The important part is the serializer's
        # load_dict can read what its dump_dict wrote.
        if ext != "toml":
            with open(file_path, "r") as f:
                loaded_native = loader(f)
            assert loaded_native == sample_dict  # Check against original for non-toml

        # Use generic load_dict - this should now work for TOML too
        loaded_dict_generic = serializer.load_dict(file_path)

        # Assert that the loaded dictionary matches the one we dumped
        assert loaded_dict_generic == dict_to_test

    def test_dump_dict_format_override(self, serializer, sample_dict, tmp_path):
        file_path = tmp_path / "data.txt"  # Use wrong extension

        # Dump as JSON
        serializer.dump_dict(sample_dict, file_path, format="json")
        with open(file_path, "r") as f:
            loaded_data = json.load(f)
        assert loaded_data == sample_dict

        # Dump as YAML
        serializer.dump_dict(sample_dict, file_path, format="yaml")
        with open(file_path, "r") as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data == sample_dict

        # Dump as TOML - Prepare a TOML-compatible dictionary first
        toml_compatible_dict = {
            "section1": {"key1": "value1", "key2": 123},  # Removed 'list'
            "section2": {"nested": {"a": True}},  # Removed 'b': None
            "top_level": sample_dict["top_level"],
            "another_list": sample_dict["another_list"],
        }
        serializer.dump_dict(toml_compatible_dict, file_path, format="toml")
        with open(file_path, "r") as f:
            loaded_data = toml.load(f)

        # The loaded data should now match the compatible dictionary we dumped
        assert loaded_data == toml_compatible_dict

    def test_dump_dict_section(self, serializer, sample_dict, tmp_path):
        file_path = tmp_path / "data_section.yaml"
        serializer.dump_dict(sample_dict, file_path, section="section1")
        loaded_data = serializer.load_dict(file_path)
        assert loaded_data == sample_dict["section1"]

    def test_dump_dict_section_in_path(self, serializer, sample_dict, tmp_path):
        file_path = tmp_path / "data_section_path.yaml"
        output_path_with_section = f"{file_path}:section2"
        serializer.dump_dict(sample_dict, output_path_with_section)
        loaded_data = serializer.load_dict(file_path)  # Load the whole file
        assert loaded_data == sample_dict["section2"]

    def test_dump_dict_section_in_path_toml(self, serializer, sample_dict, tmp_path):
        file_path = tmp_path / "data_section_path.toml"
        output_path_with_section = f"{file_path}:section2"
        serializer.dump_dict(sample_dict, output_path_with_section)
        loaded_data = serializer.load_dict(file_path)  # Load the whole file
        expected_data = {"nested": {"a": True}}  # TOML dump removes None
        assert loaded_data == expected_data

    def test_dump_dict_section_key_error(self, serializer, sample_dict, tmp_path):
        file_path = tmp_path / "data_section_error.yaml"
        with pytest.raises(KeyError, match="Section 'nonexistent' not found"):
            serializer.dump_dict(sample_dict, file_path, section="nonexistent")

    def test_dump_dict_section_in_path_key_error(self, serializer, sample_dict, tmp_path):
        file_path = tmp_path / "data_section_path_error.yaml"
        output_path_with_section = f"{file_path}:nonexistent"
        with pytest.raises(KeyError, match="Section 'nonexistent' not found"):
            serializer.dump_dict(sample_dict, output_path_with_section)

    def test_load_dict_unsupported_extension(self, serializer, tmp_path):
        file_path = tmp_path / "data.txt"
        file_path.touch()
        with pytest.raises(ValueError, match="Unsupported file extension: .txt"):
            serializer.load_dict(file_path)

    def test_dump_dict_unsupported_extension(self, serializer, sample_dict, tmp_path):
        file_path = tmp_path / "data.txt"
        with pytest.raises(ValueError, match="Unsupported file extension: .txt"):
            serializer.dump_dict(sample_dict, file_path)

    def test_dump_dict_unsupported_format(self, serializer, sample_dict, tmp_path):
        file_path = tmp_path / "data.yaml"
        with pytest.raises(ValueError, match="Unsupported format: xml"):
            serializer.dump_dict(sample_dict, file_path, format="xml")
