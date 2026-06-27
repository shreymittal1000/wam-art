"""Model adapters registry.

Re-exports the adapter hierarchy so users can do:

    from wam_art.models import BaseWAMAdapter, DummyWAMAdapter, OpenVLAAdapter
"""

from __future__ import annotations

from wam_art.models.base import BaseWAMAdapter
from wam_art.models.dummy import DummyWAMAdapter

__all__ = ["BaseWAMAdapter", "DummyWAMAdapter"]

try:
    from wam_art.models.openvla import OpenVLAAdapter  # noqa: F401
    __all__.append("OpenVLAAdapter")
except Exception:  # noqa: S110
    pass

try:
    from wam_art.models.fastwam import FastWAMAdapter  # noqa: F401
    __all__.append("FastWAMAdapter")
except Exception:  # noqa: S110
    pass

try:
    from wam_art.models.dreamzero import DreamZeroAdapter  # noqa: F401
    __all__.append("DreamZeroAdapter")
except Exception:  # noqa: S110
    pass
