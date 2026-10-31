"""
AutoND target: the pinned Crypto-TII/AutoND source and its published SPECK32/64 checkpoint.

``upstream``       hash-verified, import-isolated loading of the pinned checkout
``model_binding``  strict checkpoint binding, weight hashing, architecture, inference

Used by AutoND CE2. AutoND CE1 (``adapters/autond.py``) is completed and hash-bound by its own
package manifest, so it keeps its original loader and is not rewired to this module.
"""
