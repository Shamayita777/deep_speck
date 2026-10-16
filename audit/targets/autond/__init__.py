"""
AutoND case study (Bellini, Gerault, Hambitzer, Rossi; ePrint 2022/1467,
FSE 2024). Target repository: https://github.com/Crypto-TII/AutoND

The pinned target revision for this audit is recorded in
`target_manifest_v2.py`; no module in this package may read a different
revision of the upstream source without that manifest recording it.

Modules here are instruments and adapters only. They never modify the
upstream checkout and never write into it.
"""

PINNED_UPSTREAM_COMMIT = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"
UPSTREAM_URL = "https://github.com/Crypto-TII/AutoND"
