"""Deprecated compatibility module.

The former contents silently substituted different architectures and random
classification heads. Requested models now have dedicated strict adapters in
``models`` and are registered lazily in :mod:`models.__init__`.
"""
