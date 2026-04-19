"""
src/config.py
-------------
Configuration class that loads from YAML and provides dict-like access.
"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Iterator
import yaml

class _Namespace:
    def __init__(self, data: dict) -> None:
        object.__setattr__(self, "_data", {})
        for key, value in data.items():
            self._set(key, value)

    def _set(self, key, value):
        self._data[key] = _Namespace(value) if isinstance(value, dict) else value

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'. Available: {list(self._data.keys())}")

    def __setattr__(self, key, value):
        if key == "_data":
            object.__setattr__(self, "_data", value)
        else:
            self._set(key, value)

    def __delattr__(self, key):
        try:
            del self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __getitem__(self, key): return self._data[key]
    def __setitem__(self, key, value): self._set(key, value)
    def __delitem__(self, key): del self._data[key]
    def __contains__(self, key): return key in self._data
    def __iter__(self): return iter(self._data)
    def __len__(self): return len(self._data)
    def keys(self): return self._data.keys()
    def values(self): return (v for v in self._data.values())
    def items(self): return self._data.items()
    def get(self, key, default=None): return self._data.get(key, default)

    def to_dict(self):
        out = {}
        for key, value in self._data.items():
            out[key] = value.to_dict() if isinstance(value, _Namespace) else value
        return out

    def _merge_dict(self, override: dict) -> None:
        for key, value in override.items():
            if key in self._data and isinstance(self._data[key], _Namespace) and isinstance(value, dict):
                self._data[key]._merge_dict(value)
            else:
                self._set(key, value)

    def __repr__(self):
        inner = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"Namespace({inner})"


class Config(_Namespace):
    def __init__(self, data: dict) -> None:
        super().__init__(data)

    @classmethod
    def from_yaml(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(copy.deepcopy(data))

    def merge(self, override):
        if isinstance(override, Config):
            override = override.to_dict()
        self._merge_dict(override)
        return self

    @classmethod
    def from_yaml_with_overrides(cls, base_path, *override_paths, overrides=None):
        cfg = cls.from_yaml(base_path)
        for p in override_paths:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"Override config not found: {p}")
            with p.open("r") as fh:
                data = yaml.safe_load(fh) or {}
            cfg.merge(data)
        if overrides:
            cfg.merge(overrides)
        return cfg

    def to_yaml(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def copy(self):
        return Config.from_dict(self.to_dict())

    def __repr__(self):
        return f"Config(sections={list(self._data.keys())})"

    def __str__(self):
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, allow_unicode=True)
