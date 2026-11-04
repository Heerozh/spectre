"""
Python 3.14 evaluates annotations on demand, so a broken one only fails when something asks for
it, such as a notebook showing the help of a method. Nothing else in the tests would notice.
"""

import importlib
import inspect
import pkgutil

import spectre


def public_callables():
    names = [m.name for m in pkgutil.walk_packages(spectre.__path__, "spectre.")]
    for name in ["spectre", *names]:
        module = importlib.import_module(name)
        for attr, obj in vars(module).items():
            if getattr(obj, "__module__", None) != name:
                continue
            yield f"{name}.{attr}", obj
            if inspect.isclass(obj):
                for key, member in vars(obj).items():
                    yield f"{name}.{attr}.{key}", member


def test_signatures_resolve():
    failed = []
    for label, obj in public_callables():
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        if isinstance(obj, property):
            obj = obj.fget
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        try:
            inspect.signature(obj)
        except ValueError:
            pass  # classes without a Python signature
        except (AttributeError, NameError, TypeError) as e:
            failed.append(f"{label}: {e!r}")
    assert not failed, "\n".join(failed)
