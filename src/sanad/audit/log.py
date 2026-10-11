"""Append-only, hash-chained audit log (JSONL).

Each entry carries ``prev_hash`` and ``hash`` = SHA-256 of the canonical JSON of the entry
without its own hash (sorted keys, compact separators, UTF-8). ``verify_chain`` recomputes
every hash; editing any entry breaks the chain from that entry on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

GENESIS = "0" * 64


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def entry_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash, self._seq = GENESIS, 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    self._last_hash, self._seq = e["hash"], e["seq"] + 1

    def append(self, kind: str, ts_utc: str, payload: dict) -> dict:
        entry = {"seq": self._seq, "ts_utc": ts_utc, "kind": kind, "payload": payload, "prev_hash": self._last_hash}
        entry["hash"] = entry_hash(entry)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(canonical(entry) + "\n")
        self._last_hash, self._seq = entry["hash"], self._seq + 1
        return entry


def read_entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_chain(path_or_entries) -> tuple[bool, int | None, str]:
    """Return (ok, first_bad_seq, message)."""
    entries = read_entries(path_or_entries) if isinstance(path_or_entries, (str, Path)) else path_or_entries
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.get("prev_hash") != prev:
            return False, e.get("seq", i), f"entry {e.get('seq', i)}: prev_hash does not match previous entry"
        if entry_hash(e) != e.get("hash"):
            return False, e.get("seq", i), f"entry {e.get('seq', i)}: content hash mismatch (entry was modified)"
        prev = e["hash"]
    return True, None, f"chain intact ({len(entries)} entries)"
