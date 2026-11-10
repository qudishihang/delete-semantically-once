"""Canonical record identity and deletion-request normalisation.

The central systems distinction of this paper:

* A *request* has a key (client supplied idempotency token) and a payload.
* A *deletion effect* is the canonical set of record identities named by the payload.
* Semantic idempotence is defined over the deletion effect, never over the key.

Record identity is content addressed: ``rid = sha256(source | record_key |
version_digest)``.  Including a version digest is important: if the same primary
key is re-inserted with new content, the new record must not be mistaken for the
already deleted one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable, Sequence


def _h(*parts: str) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(p.encode("utf-8"))
        m.update(b"\x1f")
    return m.hexdigest()


def record_identity(source: str, record_key: str, version_digest: str) -> str:
    """Content-addressed identity of one record version."""
    return _h("rid/v1", source, str(record_key), version_digest)


def version_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RecordRef:
    source: str
    record_key: str
    version_digest: str

    @property
    def rid(self) -> str:
        return record_identity(self.source, self.record_key, self.version_digest)


@dataclass
class NormalisedRequest:
    """A request after canonicalisation."""
    key: str
    client: str
    rids: tuple[str, ...]          # sorted, de-duplicated
    raw_size: int                  # payload length before de-duplication
    duplicates_in_payload: int

    @cached_property
    def effect_hash(self) -> str:
        # hashing the payload is O(|A|); cache it so a request is canonicalised once
        # even though the service reads the effect identifier several times.
        return effect_hash(self.rids)

    def as_row(self) -> dict:
        return {
            "key": self.key,
            "client": self.client,
            "n_rids": len(self.rids),
            "raw_size": self.raw_size,
            "duplicates_in_payload": self.duplicates_in_payload,
            "effect_hash": self.effect_hash,
        }


def effect_hash(rids: Iterable[str]) -> str:
    """Hash of a *set* of record identities. Order-insensitive."""
    return _h("effect/v1", *sorted(set(rids)))


def normalise_request(key: str, client: str, rids: Sequence[str]) -> NormalisedRequest:
    uniq = tuple(sorted(set(rids)))
    return NormalisedRequest(key=key, client=client, rids=uniq,
                             raw_size=len(rids), duplicates_in_payload=len(rids) - len(uniq))


def delta(request_rids: Iterable[str], processed: Iterable[str]) -> tuple[str, ...]:
    """New deletion work implied by a request: A \\ H."""
    return tuple(sorted(set(request_rids) - set(processed)))


def overlap(a: Iterable[str], b: Iterable[str]) -> dict:
    sa, sb = set(a), set(b)
    inter = sa & sb
    return {
        "n_a": len(sa), "n_b": len(sb), "n_inter": len(inter),
        "jaccard": (len(inter) / len(sa | sb)) if (sa | sb) else 1.0,
        "frac_a_covered": (len(inter) / len(sa)) if sa else 1.0,
    }


def request_id_for_effect(effect: str, version: int) -> str:
    """Identity of the (effect, base-version) pair used to coalesce in-flight work."""
    return _h("coalesce/v1", effect, str(version))


def payload_digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
