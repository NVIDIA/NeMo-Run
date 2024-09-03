import dataclasses
from pathlib import Path

import pytest

import nemo_run as run
from nemo_run.io.registry import ObjectNotFoundError, _ConfigRegistry


class TestCapture:
    class DummyClass:
        def __init__(self, value):
            self.value = value

    def test_capture_as_decorator(self):
        @run.io.capture()
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

    def test_capture_as_decorator_with_cls_to_ignore(self):
        class IgnoredClass:
            def __init__(self, value):
                self.value = value

        @run.io.capture(cls_to_ignore={IgnoredClass})
        def create_objects():
            obj1 = self.DummyClass(1)
            obj2 = IgnoredClass(2)
            return obj1, obj2

        obj1, obj2 = create_objects()

        assert isinstance(run.io.get(obj1), run.Config)
        with pytest.raises(ObjectNotFoundError):
            run.io.get(obj2)

    def test_nested_capture(self):
        with run.io.capture():
            obj1 = self.DummyClass(1)
            with run.io.capture():
                obj2 = self.DummyClass(2)

        assert isinstance(run.io.get(obj1), run.Config)
        assert isinstance(run.io.get(obj2), run.Config)
        assert run.io.get(obj1).value == 1
        assert run.io.get(obj2).value == 2

    def test_capture_exception_handling(self):
        class TestException(Exception):
            pass

        with pytest.raises(TestException):
            with run.io.capture():
                obj = self.DummyClass(42)
                raise TestException("Test exception")

        # The object should still be captured despite the exception
        assert isinstance(run.io.get(obj), run.Config)
        assert run.io.get(obj).value == 42

    def test_capture_nested_objects(self):
        class NestedClass:
            def __init__(self, value):
                self.value = value

        class OuterClass:
            def __init__(self, nested):
                self.nested = nested

        with run.io.capture():
            nested = NestedClass(42)
            outer = OuterClass(nested)

        assert isinstance(run.io.get(outer), run.Config)
        assert isinstance(run.io.get(outer).nested, run.Config)
        assert run.io.get(outer).nested.value == 42

    def test_capture_complex_arguments(self):
        class ComplexClass:
            def __init__(self, list_arg, dict_arg):
                self.list_arg = list_arg
                self.dict_arg = dict_arg

        with run.io.capture():
            obj = ComplexClass([1, 2, 3], {"a": 1, "b": 2})

        cfg = run.io.get(obj)
        assert isinstance(cfg, run.Config)
        assert cfg.list_arg == [1, 2, 3]
        assert cfg.dict_arg == {"a": 1, "b": 2}

    def test_capture_callable_arguments(self):
        def dummy_func():
            pass

        class CallableClass:
            def __init__(self, func):
                self.func = func

        with run.io.capture():
            obj = CallableClass(dummy_func)

        cfg = run.io.get(obj)
        assert isinstance(cfg, run.Config)
        assert cfg.func == dummy_func

    # TODO: Fix this test
    # def test_capture_path_arguments(self):
    #     class PathClass:
    #         def __init__(self, path):
    #             self.path = path

    #     with run.io.capture():
    #         obj = PathClass(Path("/tmp/test"))

    #     cfg = run.io.get(obj)
    #     assert isinstance(cfg, run.Config)
    #     assert isinstance(cfg.path, run.Config)
    #     assert cfg.path.args[0] == "/tmp/test"

    def test_capture_multiple_objects(self):
        class ClassA:
            def __init__(self, value):
                self.value = value

        class ClassB:
            def __init__(self, value):
                self.value = value

        with run.io.capture():
            obj_a = ClassA(1)
            obj_b = ClassB("test")

        assert isinstance(run.io.get(obj_a), run.Config)
        assert isinstance(run.io.get(obj_b), run.Config)
        assert run.io.get(obj_a).value == 1
        assert run.io.get(obj_b).value == "test"

    def test_capture_unsupported_type(self):
        class UnsupportedClass:
            def __init__(self):
                pass

        class TestClass:
            def __init__(self, unsupported):
                self.unsupported = unsupported

        unsupported = UnsupportedClass()

        with pytest.raises(ValueError, match="Unable to convert object of type"):
            with run.io.capture():
                TestClass(unsupported)

    # TODO: fix
    # def test_capture_with_inheritance(self):
    #     class BaseClass:
    #         def __init__(self, base_value):
    #             self.base_value = base_value

    #     class DerivedClass(BaseClass):
    #         def __init__(self, base_value, derived_value):
    #             super().__init__(base_value)
    #             self.derived_value = derived_value

    #     with run.io.capture():
    #         obj = DerivedClass(1, "test")

    #     cfg = run.io.get(obj)
    #     assert isinstance(cfg, run.Config)
    #     assert cfg.base_value == 1
    #     assert cfg.derived_value == "test"

    def test_capture_with_default_arguments(self):
        class DefaultArgClass:
            def __init__(self, arg1, arg2="default"):
                self.arg1 = arg1
                self.arg2 = arg2

        with run.io.capture():
            obj1 = DefaultArgClass(1)
            obj2 = DefaultArgClass(2, "custom")

        cfg1 = run.io.get(obj1)
        cfg2 = run.io.get(obj2)

        assert cfg1.arg1 == 1
        assert cfg1.arg2 == "default"
        assert cfg2.arg1 == 2
        assert cfg2.arg2 == "custom"

    def test_capture_exception_handling(self):
        class TestException(Exception):
            pass

        with pytest.raises(TestException):
            with run.io.capture():
                obj = self.DummyClass(42)
                raise TestException("Test exception")

        # The object should still be captured despite the exception
        assert isinstance(run.io.get(obj), run.Config)
        assert run.io.get(obj).value == 42

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
