"""Patch the __str__ method of functools.partial to make it suitable for serialization."""

import functools


class partial(functools.partial):
    def __str__(self):
        return f"functools.partial(<function {self.func.__name__}>, {', '.join(f'{k}={v}' for k, v in self.keywords.items())})"
