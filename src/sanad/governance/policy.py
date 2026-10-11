"""Governance policy table (CLAUDE.md §10): autonomy follows reversibility and impact.

Tiers: AUTO (SANAD does it, logged), APPROVAL (operator must approve), IMMEDIATE
(immediate human attention). SANAD never actuates the plant: every action below is an
advisory or a data-gating action on the trusted data stream.
"""

from __future__ import annotations

TIERS = ["AUTO", "APPROVAL", "IMMEDIATE"]

POLICY_TABLE = [
    # situation, automatic, needs approval, immediate human
    ("Short data gap", "impute, label ESTIMATED", "", ""),
    ("Sensor drift/bias, conf >= 0.85", "quarantine + substitute estimate + open ticket", "recalibration commit", ""),
    ("Sensor drift/bias, conf 0.60-0.85", "label SUSPECT only", "quarantine", ""),
    ("Suspected falsification", "hold data as SUSPECT", "any conclusion (AI never accuses)", ""),
    ("Genuine overload / grid event", "notify, no data correction", "", "if sustained or beyond thermal limit"),
    ("Bearing degradation", "maintenance advisory", "shutdown scheduling", ""),
    ("UNCERTAIN", "log + queue", "human decides", ""),
    ("Threshold / model change, importing a Pattern Card", "", "always", ""),
    (">= 2 critical events or multiple sensors lost", "", "", "always"),
]


def tighten(tier: str) -> str:
    """One step less autonomy (adaptive autonomy)."""
    return TIERS[min(len(TIERS) - 1, TIERS.index(tier) + 1)]
