"""Typed reads of checkpoint config fields shared by the architecture adapters."""

from __future__ import annotations

from math import isfinite
import struct


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def _f32(value, name):
    if type(value) not in (float, int):
        raise ValueError(f"{name}: expected a real value")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{name}: not representable as F32") from error
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name}: expected positive finite F32")
    return result


def _fixed(config, key, value, label):
    if key in config and (
        config[key] != value
        or isinstance(value, bool)
        and type(config[key]) is not bool
    ):
        raise ValueError(f"{label}.{key}: expected {value!r}, got {config[key]!r}")


def _rope_source(raw: dict, label: str) -> dict:
    rope = {}
    for field in ("rope_scaling", "rope_parameters"):
        value = raw.get(field)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{label}.{field}: expected an object")
        for alias in ("type", "rope_type"):
            _fixed(value, alias, "default", label + "." + field)
        if "factor" in value and _f32(value["factor"], label + ".factor") != 1.0:
            raise ValueError(f"{label}: scaled RoPE is not implemented")
        for key, item in value.items():
            if key in rope and rope[key] != item:
                raise ValueError(f"{label}: conflicting RoPE field {key}")
            rope[key] = item
    return rope
