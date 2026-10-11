"""Energy Noosphere — Pattern Card schema (Phase 1: schema + protocol only, no network).

A Pattern Card is what one SANAD node may share with others: an abstract, anonymised
signature of a diagnosed pattern (which relations broke / held, typical magnitudes,
recommended governance) — never raw measurements, never site identity. Importing a card
always requires operator approval (governance table: "importing a Pattern Card").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "0.1"

# What is shared, what stays local, and why (Phase 1 protocol table)
SHARING_TABLE = [
    ("Relation signature (violated / held relations)", "shared", "Physics is site-independent; lets other nodes recognise the pattern"),
    ("Normalised magnitudes (z-score ranges, % disagreement)", "shared", "Relative values only; no absolute plant loads"),
    ("Verdict class, subtype, confidence calibration summary", "shared", "Lets peers weigh the card's reliability"),
    ("Recommended governance tier and actions", "shared", "Spreads good practice (e.g. quarantine + substitute)"),
    ("Equipment class (e.g. 'induction motor 50-150 kW')", "shared", "Coarse class only, never an asset tag"),
    ("Raw time series, kWh, timestamps", "local", "Commercially sensitive; not needed to reuse a pattern"),
    ("Site, company, node identity, sensor IDs", "local", "Privacy and security; cards are anonymised"),
    ("Operator names and override reasons", "local", "Personal data; only aggregate override rates may be shared"),
    ("Thresholds and calibration constants", "local", "Site-specific; a peer must re-learn them on its own normal data"),
]


@dataclass
class PatternCard:
    card_id: str
    schema_version: str
    verdict_class: str
    subtype: str
    equipment_class: str
    relations_violated: list[str]
    relations_held: list[str]
    magnitude_hints: dict = field(default_factory=dict)
    recommended_tier: str = "APPROVAL"
    recommended_actions: list[str] = field(default_factory=list)
    evidence_cases: int = 0
    calibrated_accuracy: float | None = None
    provenance: str = "synthetic"  # synthetic | field
    signature: str = ""

    def compute_signature(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "signature"}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_json(self) -> str:
        self.signature = self.compute_signature()
        return json.dumps(asdict(self), indent=2)


JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SANAD Pattern Card",
    "type": "object",
    "required": ["card_id", "schema_version", "verdict_class", "subtype", "equipment_class",
                 "relations_violated", "relations_held", "recommended_tier", "signature"],
    "properties": {
        "card_id": {"type": "string"},
        "schema_version": {"const": SCHEMA_VERSION},
        "verdict_class": {"type": "string"},
        "subtype": {"type": "string"},
        "equipment_class": {"type": "string"},
        "relations_violated": {"type": "array", "items": {"type": "string", "pattern": "^R[0-9]+[A-Z_]*$"}},
        "relations_held": {"type": "array", "items": {"type": "string"}},
        "magnitude_hints": {"type": "object"},
        "recommended_tier": {"enum": ["AUTO", "APPROVAL", "IMMEDIATE"]},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "evidence_cases": {"type": "integer", "minimum": 0},
        "calibrated_accuracy": {"type": ["number", "null"]},
        "provenance": {"enum": ["synthetic", "field"]},
        "signature": {"type": "string", "description": "SHA-256 of the canonical card body"},
    },
}


def example_card() -> PatternCard:
    """Illustrative example (values are illustrative, not measured results)."""
    return PatternCard(
        card_id="PC-ILLUSTRATIVE-0001", schema_version=SCHEMA_VERSION, verdict_class="SENSOR_FAULT", subtype="DRIFT",
        equipment_class="induction motor 50-150 kW, feeder meter + relay CT", relations_violated=["R3", "R4"],
        relations_held=["R1", "R2", "R6"], magnitude_hints={"meter_vs_relay_pct": "illustrative: -2 .. -6"},
        recommended_tier="AUTO", recommended_actions=["quarantine meter CT", "substitute relay-based estimate",
                                                      "recalibration commit (approval)"],
        evidence_cases=0, calibrated_accuracy=None, provenance="synthetic")
