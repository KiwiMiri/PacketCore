"""Feature schema definition and versioning.

All feature names, dtypes, and flags are declared here so that every
pipeline stage uses the same agreed-upon contract.

Increment SCHEMA_VERSION whenever the feature set changes so that saved
artefacts remain traceable to the schema they were produced with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

SCHEMA_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Per-field descriptor
# ---------------------------------------------------------------------------


@dataclass
class FieldSpec:
    """Specification for a single feature field.

    Attributes
    ----------
    name:
        Canonical field name used throughout the pipeline.
    dtype:
        NumPy-compatible dtype string, e.g. ``"float32"``.
    requires_payload:
        When ``True`` this field is only valid when ``is_payload_visible``
        is ``True`` (i.e. the packet payload has not been encrypted / redacted).
        Fields with ``requires_payload=True`` **must** be gated at extraction
        time so that encrypted protocols (e.g. OPC UA over TLS) never rely on
        them.
    description:
        Human-readable description of the field.
    """

    name: str
    dtype: str = "float32"
    requires_payload: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Schema class
# ---------------------------------------------------------------------------


@dataclass
class FeatureSchema:
    """Container for the full feature schema.

    Parameters
    ----------
    version:
        Schema version string (should match :data:`SCHEMA_VERSION`).
    fields:
        Ordered list of :class:`FieldSpec` objects describing every feature.
    """

    version: str = SCHEMA_VERSION
    fields: List[FieldSpec] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def names(self) -> List[str]:
        """Return ordered list of field names."""
        return [f.name for f in self.fields]

    def payload_gated_names(self) -> List[str]:
        """Return names of fields that require payload visibility."""
        return [f.name for f in self.fields if f.requires_payload]

    def metadata_only_names(self) -> List[str]:
        """Return names of fields that do NOT require payload visibility."""
        return [f.name for f in self.fields if not f.requires_payload]

    def dtype_map(self) -> Dict[str, str]:
        """Return mapping of field name → dtype."""
        return {f.name: f.dtype for f in self.fields}

    def index_of(self, name: str) -> int:
        """Return the positional index of *name* in the schema."""
        for i, f in enumerate(self.fields):
            if f.name == name:
                return i
        raise KeyError(f"Field '{name}' not found in schema version {self.version}")

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary (JSON-safe)."""
        return {
            "version": self.version,
            "fields": [
                {
                    "name": f.name,
                    "dtype": f.dtype,
                    "requires_payload": f.requires_payload,
                    "description": f.description,
                }
                for f in self.fields
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSchema":
        """Deserialise from a plain dictionary."""
        return cls(
            version=d.get("version", SCHEMA_VERSION),
            fields=[FieldSpec(**fdict) for fdict in d.get("fields", [])],
        )


# ---------------------------------------------------------------------------
# Default schema (v1.0.0)
# ---------------------------------------------------------------------------

#: Canonical schema used by all pipeline components unless overridden.
DEFAULT_SCHEMA: FeatureSchema = FeatureSchema(
    version=SCHEMA_VERSION,
    fields=[
        # --- Metadata / statistical features (always available) ---
        FieldSpec("timestamp", "float64", False, "Unix epoch timestamp of the packet/flow record"),
        FieldSpec("src_ip_int", "int64", False, "Source IP address encoded as integer"),
        FieldSpec("dst_ip_int", "int64", False, "Destination IP address encoded as integer"),
        FieldSpec("src_port", "int32", False, "Source TCP/UDP port"),
        FieldSpec("dst_port", "int32", False, "Destination TCP/UDP port"),
        FieldSpec("protocol", "int32", False, "IP protocol number (e.g. 6=TCP, 17=UDP)"),
        FieldSpec("pkt_len", "float32", False, "Packet length in bytes"),
        FieldSpec("ip_ttl", "float32", False, "IP TTL field"),
        FieldSpec("tcp_flags", "int32", False, "TCP flags bitmask (0 for non-TCP)"),
        FieldSpec("flow_duration", "float32", False, "Flow duration in seconds"),
        FieldSpec("flow_pkt_count", "float32", False, "Number of packets in the flow"),
        FieldSpec("flow_byte_count", "float32", False, "Total bytes in the flow"),
        FieldSpec("flow_pkt_rate", "float32", False, "Packets per second"),
        FieldSpec("flow_byte_rate", "float32", False, "Bytes per second"),
        FieldSpec("inter_arrival_mean", "float32", False, "Mean inter-arrival time (seconds)"),
        FieldSpec("inter_arrival_std", "float32", False, "Std-dev of inter-arrival time"),
        FieldSpec("pkt_len_mean", "float32", False, "Mean packet length in flow"),
        FieldSpec("pkt_len_std", "float32", False, "Std-dev of packet length in flow"),
        FieldSpec("pkt_len_min", "float32", False, "Min packet length in flow"),
        FieldSpec("pkt_len_max", "float32", False, "Max packet length in flow"),
        FieldSpec("is_payload_visible", "int8", False, "1 if DPI payload is available, else 0"),
        # --- OPC UA specific (metadata-level, no payload required) ---
        FieldSpec("opcua_msg_type", "int32", False, "OPC UA message type enum (0 if N/A)"),
        FieldSpec("opcua_security_mode", "int32", False, "OPC UA security mode (0=None,1=Sign,2=SignEncrypt)"),
        # --- Payload-derived features (only valid when is_payload_visible=1) ---
        FieldSpec("payload_entropy", "float32", True, "Shannon entropy of payload bytes"),
        FieldSpec("payload_len", "float32", True, "Raw payload length in bytes"),
        FieldSpec("dpi_app_proto", "int32", True, "DPI-detected application protocol enum"),
    ],
)
