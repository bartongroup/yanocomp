import dataclasses
import click


def dynamic_dataclass(cls_name, bases=None, **kwargs):
    fields = [
        (kw, type(val), val) for kw, val in kwargs.items()
    ]
    dc = dataclasses.make_dataclass(cls_name, fields, bases=bases)
    return dc()


def make_dataclass_decorator(cls_name, bases=None):
    def _dataclass_decorator(cmd):
        def _make_dataclass(**cli_kwargs):
            return cmd(dynamic_dataclass(cls_name, bases=bases, **cli_kwargs))
        # pass docstring
        _make_dataclass.__doc__ = cmd.__doc__
        return _make_dataclass
    return _dataclass_decorator