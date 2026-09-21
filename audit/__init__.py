"""
CipherMind audit framework.

Four audit pillars (per the governing methodology, Sections 3.5-3.8):
    audit.implementation  - Implementation Integrity
    audit.dataset          - Dataset Integrity
    audit.experimental      - Experimental Validity
    audit.cryptography      - Cryptographic Evidence (CE1-CE4)

One cross-pillar integration layer (NOT a fifth pillar):
    audit.integration

Shared, dimension-agnostic infrastructure used by all of the above:
    audit.common

Import direction is one-way: audit.common is imported by everything;
audit.integration imports from the four pillars; the four pillars never
import from audit.integration or from each other.
"""
