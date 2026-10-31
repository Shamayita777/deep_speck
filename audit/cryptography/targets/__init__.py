"""
Target layer: what is specific to one published system (cipher implementation, upstream source,
checkpoint binding, inference), shared by every experiment on that system.

Targets reference their sources where they already are. Nothing is copied or moved:

* ``targets.gohr``   -> the Gohr code at ``audit/cryptography/gohr/`` and ``Archive/best5depth10.h5``
* ``targets.autond`` -> the pinned AutoND checkout at ``audit/cryptography/autond_ce1/AutoND-upstream/``
"""
