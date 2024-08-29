import dataclasses

import pytest

import nemo_run as run
from nemo_run.io.registry import ObjectNotFoundError, _ConfigRegistry


class TestCapture:
    class DummyClass:
        def __init__(self, value):
            self.value = value

    def test_capture_as_decorator(self):
        @run.io.capture
        def create_object():
            return self.DummyClass(42)

        obj = create_object()
        assert isinstance(obj, self.DummyClass)
        assert obj.value == 42

        cfg = run.io.get(obj)
        assert isinstance(cfg, run.Config)
        assert cfg.value == 42

    def test_capture_as_context_manager(self):
        with run.io.capture():
            obj = self.DummyClass(42)

        assert isinstance(obj, self.DummyClass)
        assert obj.value == 42

        cfg = run.io.get(obj)
        assert isinstance(cfg, run.Config)
        assert cfg.value == 42

    def test_capture_with_cls_to_ignore(self):
        class IgnoredClass:
            def __init__(self, value):
                self.value = value

        with run.io.capture(cls_to_ignore={IgnoredClass}):
            obj1 = self.DummyClass(1)
            obj2 = IgnoredClass(2)

        assert isinstance(run.io.get(obj1), run.Config)
        with pytest.raises(ObjectNotFoundError):
            run.io.get(obj2)


class TestReinit:
    def test_simple(self):
        class DummyClass:
            def __init__(self, value):
                self.value = value

        cfg = run.Config(DummyClass, value=42)
        instance = run.build(cfg)

        new_instance = run.io.reinit(instance)
        assert isinstance(new_instance, DummyClass)
        assert new_instance.value == 42
        assert new_instance is not instance

    def test_reinit_not_registered(self):
        class DummyClass:
            pass

        instance = DummyClass()

        with pytest.raises(ObjectNotFoundError):
            run.io.reinit(instance)

    def test_reinit_dataclass(self):
        """Test reinitializing a dataclass instance."""

        @dataclasses.dataclass
        class DummyDataClass:
            value: int
            name: str

        instance = DummyDataClass(value=42, name="test")
        new_instance = run.io.reinit(instance)

        assert isinstance(new_instance, DummyDataClass)
        assert new_instance.value == 42
        assert new_instance.name == "test"
        assert new_instance is not instance


class TestIOCleanup:
    @pytest.fixture
    def registry(self):
        return _ConfigRegistry()

    def test_cleanup_removes_garbage_collected_objects(self, registry):
        class DummyObject:
            pass

        obj1 = DummyObject()
        obj2 = DummyObject()
        cfg1 = run.Config(DummyObject)
        cfg2 = run.Config(DummyObject)

        obj1_id = id(obj1)  # Store the id before deleting the object
        registry.register(obj1, cfg1)
        registry.register(obj2, cfg2)

        assert len(registry) == 2

        del obj1  # Make obj1 eligible for garbage collection
        registry.cleanup()

        assert len(registry) == 1
        with pytest.raises(ObjectNotFoundError):
            registry.get_by_id(obj1_id)  # Use a new method to get by id
        assert registry.get(obj2) == cfg2

    def test_cleanup_keeps_live_objects(self, registry):
        class DummyObject:
            pass

        obj = DummyObject()
        cfg = run.Config(DummyObject)

        registry.register(obj, cfg)
        registry.cleanup()

        assert len(registry) == 1
        assert registry.get(obj) == cfg

    def test_cleanup_with_empty_registry(self, registry):
        registry.cleanup()
        assert len(registry) == 0

    def test_cleanup_multiple_times(self, registry):
        class DummyObject:
            pass

        obj1 = DummyObject()
        obj2 = DummyObject()
        cfg1 = run.Config(DummyObject)
        cfg2 = run.Config(DummyObject)

        registry.register(obj1, cfg1)
        registry.register(obj2, cfg2)

        assert len(registry) == 2

        del obj1  # Make obj1 eligible for garbage collection
        registry.cleanup()
        assert len(registry) == 1

        registry.cleanup()  # Second cleanup should not change anything
        assert len(registry) == 1

        del obj2  # Make obj2 eligible for garbage collection
        registry.cleanup()
        assert len(registry) == 0

    def test_cleanup_after_reregistration(self, registry):
        class DummyObject:
            pass

        obj = DummyObject()
        cfg1 = run.Config(DummyObject)
        cfg2 = run.Config(DummyObject)

        registry.register(obj, cfg1)
        registry.register(obj, cfg2)  # Re-register with a new config

        registry.cleanup()

        assert len(registry) == 1
        assert registry.get(obj) == cfg2

    def test_cleanup_stress_test(self, registry):
        class DummyObject:
            pass

        objects = []
        for _ in range(10000):
            obj = DummyObject()
            objects.append(obj)
            registry.register(obj, run.Config(DummyObject))

        assert len(registry) == 10000

        # Delete all objects
        del objects

        # Force garbage collection
        import gc

        gc.collect()

        registry.cleanup()
        assert len(registry) == 1
