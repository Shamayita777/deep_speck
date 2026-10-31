"""
Shared core of the cryptography audit framework (additive layer, introduced 2026-10-31).

The modules here own experiment-independent infrastructure: the evidence lifecycle (journal,
state machine, crash-safe artifact commits), hypothesis-direction semantics, numerically honest
p-values, production approval records, capability interfaces and the registry of production
routes and frozen files.

The older framework modules at the package root (``results.py``, ``statistics.py``,
``test/``, ``adapters/base.py``, ``engine.py`` and others) are bound by
``PACKAGE_MANIFEST.sha256`` and by downstream evidence, so they are kept byte-identical and are
not edited here. ``core/registry.py`` records which of them production actually uses.
"""
