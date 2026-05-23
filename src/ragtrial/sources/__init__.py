"""Source registry — all domains, assembled in one place.

`SOURCES` maps domain name -> Source. The capability registry, the preprocess
CLI, and the vector-store build CLI all iterate this dict, so adding a domain
is: create `sources/<domain>/`, import its `source` here, add to the list.
"""

from __future__ import annotations

from ragtrial.sources.base import Source, save_docs
from ragtrial.sources.dukcapil import source as dukcapil_source
from ragtrial.sources.opd import source as opd_source

_ALL: list[Source] = [dukcapil_source, opd_source]

SOURCES: dict[str, Source] = {s.name: s for s in _ALL}

__all__ = ["Source", "SOURCES", "save_docs"]
