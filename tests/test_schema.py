"""Tests for feature schema."""

from __future__ import annotations

import pytest

from packetcore.features.schema import (
    DEFAULT_SCHEMA,
    SCHEMA_VERSION,
    FeatureSchema,
    FieldSpec,
)


def test_schema_version_string():
    assert isinstance(SCHEMA_VERSION, str)
    parts = SCHEMA_VERSION.split(".")
    assert len(parts) == 3


def test_default_schema_has_fields():
    assert len(DEFAULT_SCHEMA.fields) > 0


def test_names_returns_list_of_strings():
    names = DEFAULT_SCHEMA.names()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)


def test_payload_gated_names_are_subset():
    gated = set(DEFAULT_SCHEMA.payload_gated_names())
    all_names = set(DEFAULT_SCHEMA.names())
    assert gated.issubset(all_names)


def test_metadata_only_names_are_subset():
    meta = set(DEFAULT_SCHEMA.metadata_only_names())
    all_names = set(DEFAULT_SCHEMA.names())
    assert meta.issubset(all_names)


def test_gated_and_metadata_partition():
    gated = set(DEFAULT_SCHEMA.payload_gated_names())
    meta = set(DEFAULT_SCHEMA.metadata_only_names())
    all_names = set(DEFAULT_SCHEMA.names())
    assert gated.isdisjoint(meta)
    assert gated | meta == all_names


def test_index_of_known_field():
    idx = DEFAULT_SCHEMA.index_of("timestamp")
    assert isinstance(idx, int)
    assert idx >= 0


def test_index_of_unknown_field_raises():
    with pytest.raises(KeyError):
        DEFAULT_SCHEMA.index_of("nonexistent_field_xyz")


def test_serialisation_round_trip():
    d = DEFAULT_SCHEMA.to_dict()
    restored = FeatureSchema.from_dict(d)
    assert restored.version == DEFAULT_SCHEMA.version
    assert restored.names() == DEFAULT_SCHEMA.names()


def test_dtype_map():
    dm = DEFAULT_SCHEMA.dtype_map()
    assert "timestamp" in dm
    assert dm["timestamp"] == "float64"


def test_is_payload_visible_not_gated():
    # The flag field itself should NOT require payload
    meta = DEFAULT_SCHEMA.metadata_only_names()
    assert "is_payload_visible" in meta


def test_payload_entropy_is_gated():
    gated = DEFAULT_SCHEMA.payload_gated_names()
    assert "payload_entropy" in gated
