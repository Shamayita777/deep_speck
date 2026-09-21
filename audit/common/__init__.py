"""
audit.common — shared, dimension-agnostic infrastructure.

This package contains NO knowledge of Gohr, Speck, neural distinguishers,
or any other specific cryptanalytic case study. It is consumed by all
four audit pillars (implementation, dataset, experimental, cryptographic)
and by the integration layer, but never the reverse: nothing here may
import from audit.dataset, audit.experimental, audit.cryptography,
audit.implementation, or audit.integration.

Per the governing methodology (Section 3.1.2), the audit methodology is
"designed to operate independently of the underlying cryptographic
primitive, the neural network architecture, the implementation language,
the machine learning framework, and the dataset generation pipeline."
This package is where that independence is actually enforced in code,
not just asserted in prose.
"""
