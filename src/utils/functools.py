"""Patch the __str__ method of functools.partial to make it suitable for serialization."""

import functools


class partial(functools.partial):
    def __repr__(self):
        keywords = ", ".join(f"{k}={v}" for k, v in self.keywords.items())
        keywords = f", {keywords}" if keywords else ""

        return f"functools.partial(<function {self.func.__name__}>{keywords})"
