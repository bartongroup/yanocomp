import dataclasses
import click


def make_dataclass_decorator(cls_name, **kwargs):
    def _dataclass_decorator(cmd):
        @click.pass_context
        def _make_dataclass(ctx, **cli_kwargs):
            fields = [
                (kw, type(val), val) for kw, val in cli_kwargs.items()
            ]
            dc = dataclasses.make_dataclass(cls_name, fields, **kwargs)
            return ctx.invoke(cmd, dc())
        # pass docstring
        _make_dataclass.__doc__ = cmd.__doc__
        return _make_dataclass
    return _dataclass_decorator