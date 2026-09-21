"""
Gohr/Speck32/64 adapter for Experimental Validity.

Contains all Gohr-specific implementation details: dataset generation,
model construction, training, evaluation, the frozen baseline
configuration, and the representation perturbation used by
H-EV-REPRESENTATION.

Independence note (see docs/independence.md): this adapter reuses only
the task-defining Speck cipher primitives (gohr.speck, ported verbatim
from archive/speck.py with attribution) and the unmodified architecture
(gohr.model, ported verbatim from archive/train_nets.py). It does NOT
import audit.dataset.adapters.gohr or audit.cryptography.gohr, because
those modules' defaults and undeclared choices (round count, depth,
shuffle behavior) are among the things EV investigates - importing them
would make EV's baseline silently inherit one side's choices.
"""
