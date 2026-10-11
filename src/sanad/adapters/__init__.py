"""Real-data adapters (Phase 2). Interfaces + TODO stubs only.

Every adapter must ultimately produce the exact same output schema as the synthetic
generator (see ``sanad.schema``), so the rest of the platform is source-agnostic.
"""

from sanad.adapters.base import DataSourceAdapter

__all__ = ["DataSourceAdapter"]
