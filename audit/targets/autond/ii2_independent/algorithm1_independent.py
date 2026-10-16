"""Independent reimplementation of AutoND's evolutionary optimizer (audit stage II-2).

Derived from the paper text alone; see PROVENANCE.md.

Numbering: every "Alg. 1", line label, section and definition number in this
file refers to the PDF supplied to II-2. In that PDF the evolutionary
optimizer is Algorithm 1 (§5.2) and Gohr's optimizer is Algorithm 2
(Appendix A). Other revisions number them differently (AMBIGUITY_LOG.md#A-031).
Line labels L1-L19 are defined in SPECIFICATION.md §3.

Public API
    search(cipher, *, seed, **params) -> dict        the audit contract (brief §7)
    bias_score(cipher, delta, rounds, samples, rng)   Definition 2 (§5.1)
    generate_offspring(population, n, M, rng)         Alg. 1 L5-L15
    PAPER_DEFAULTS, PAPER_REDUCED                     read-only parameter sets
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Callable, Optional, Sequence

import numpy as np

from .cipher_interface import Cipher, int_to_bits, validate_cipher

__all__ = [
    "PAPER_DEFAULTS",
    "PAPER_REDUCED",
    "STARTING_ROUND",
    "ROUND_WEIGHTINGS",
    "RoundResult",
    "bias_score",
    "generate_offspring",
    "search",
]

Scorer = Callable[[Cipher, int, int, int, np.random.Generator], float]

# --------------------------------------------------------------------------
# Parameters taken from the paper. Read-only mappings: no module-level
# mutable state.
# --------------------------------------------------------------------------
PAPER_DEFAULTS = MappingProxyType({
    "initial_population": 1024,   # Alg. 1 L1; §5.2 (initial population of 1024)
    "population_size": 32,        # Alg. 1 L3, L17; §5.2 (32 best kept per generation)
    "generations": 50,            # Alg. 1 L4, exclusive bound (A-002); §8 (50 generations)
    "mutation_probability": 1.0,  # §5.2 (M = 1)
    "samples": 10_000,            # §5.2 (t = 10^4)
    "threshold": 0.01,            # §5.2 (T_b = 0.01); §5.4
})

PAPER_REDUCED = MappingProxyType({
    "samples": 1_000,             # §5.4 reduced set (t = 10^3)
    "threshold": 0.05,            # §5.4 reduced set (T_b = 0.05)
    "generations": 10,            # §5.4 reduced set (10 generations)
})

STARTING_ROUND = 1                # §5.2: optimizer is run iteratively from round 1

# A-017: "linear" (w_r = r) is II-2's own choice; the paper's prose asks for
# weights increasing with r but states none. "uniform" is the equation as
# printed in §5.2.
ROUND_WEIGHTINGS = ("linear", "uniform")

_PAPER_SOURCES = MappingProxyType({
    "initial_population": "Alg. 1 L1 [given PDF]; §5.2",
    "population_size": "Alg. 1 L3, L17 [given PDF]; §5.2",
    "generations": "Alg. 1 L4 [given PDF], exclusive bound (A-002); §8",
    "mutation_probability": "§5.2",
    "samples": "§5.2",
    "threshold": "§5.2; §5.4",
    "starting_round": "§5.2",
    "elitism": "Alg. 1 L17 [given PDF] (A-006)",
})
_REDUCED_SOURCE = "§5.4 (reduced parameter set)"
# round_weighting is deliberately ABSENT: w_r = r is an II-2 construction, not
# a paper value, so it must never be reported in paper_parameters (A-017).
_EXTRA_PAPER_VALUES = MappingProxyType({
    "starting_round": STARTING_ROUND,
    "elitism": False,
})

_BIAS_CENTRE = 0.5   # Definition 2: per-bit bias is |0.5 - frequency|

# RNG stream layout. NOT IN PAPER: chosen as spawn_key-addressed streams,
# see SPECIFICATION.md §8.
_STREAM_ROUND = 1
_STREAM_RESCORE = 2
_SUB_SEARCH = 0
_SUB_SCORE = 1

_BYTE_VALUES = 256   # representation constant: uniform bytes for RandomInt(0, 2^n - 1)
_BITS_PER_BYTE = 8   # representation constant


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------
def _check_int(name: str, value, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_real(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value}")
    return out


def _check_bool(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class _Config:
    initial_population: int
    population_size: int
    generations: int
    mutation_probability: float
    samples: int
    threshold: float
    round_weighting: str
    deduplicate: bool
    elitism: bool


def _build_config(
    *,
    initial_population,
    population_size,
    generations,
    mutation_probability,
    samples,
    threshold,
    round_weighting,
    deduplicate,
    elitism,
) -> _Config:
    p0 = _check_int("initial_population", initial_population, 1)
    p = _check_int("population_size", population_size, 2)  # pairs need >= 2 (L6-L7)
    if p > p0:
        raise ValueError(
            f"population_size ({p}) cannot exceed initial_population ({p0}); "
            "L3 keeps the first population_size of the initial population"
        )
    g = _check_int("generations", generations, 0)
    m = _check_real("mutation_probability", mutation_probability)
    if not 0.0 <= m <= 1.0:
        raise ValueError(f"mutation_probability must be in [0, 1], got {m}")
    t = _check_int("samples", samples, 1)
    tb = _check_real("threshold", threshold)
    if tb < 0.0:
        raise ValueError(f"threshold must be >= 0, got {tb}")
    if round_weighting not in ROUND_WEIGHTINGS:
        raise ValueError(f"round_weighting must be one of {ROUND_WEIGHTINGS}, got {round_weighting!r}")
    return _Config(
        initial_population=p0,
        population_size=p,
        generations=g,
        mutation_probability=m,
        samples=t,
        threshold=tb,
        round_weighting=round_weighting,
        deduplicate=_check_bool("deduplicate", deduplicate),
        elitism=_check_bool("elitism", elitism),
    )


def _generator(seed: int, key: tuple) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=key)))


# --------------------------------------------------------------------------
# Definition 2: bias score
# --------------------------------------------------------------------------
def _encrypt_checked(cipher: Cipher, plaintexts: np.ndarray, keys: np.ndarray, rounds: int) -> np.ndarray:
    out = np.asarray(cipher.encrypt(plaintexts, keys, rounds))
    expected = (plaintexts.shape[0], cipher.output_bits)
    if out.shape != expected:
        raise ValueError(f"cipher.encrypt returned shape {out.shape}, expected {expected}")
    if out.dtype != np.bool_:
        if not np.issubdtype(out.dtype, np.integer):
            raise TypeError(f"cipher.encrypt must return integer or bool bits, got {out.dtype}")
        if ((out < 0) | (out > 1)).any():
            raise ValueError("cipher.encrypt must return only 0/1 values")
    return out.astype(np.uint8, copy=False)


def bias_score(cipher: Cipher, delta: int, rounds: int, samples: int, rng: np.random.Generator) -> float:
    """Bias score b~^t_r(delta), Definition 2 (§5.1), at ``rounds`` rounds.

    Mean over output bits j of |0.5 - (fraction of samples whose output
    difference has bit j set)|. Mean rather than sum (A-013); absolute value
    (A-014). Fresh X_i and K_i for every call (A-015). Range [0, 0.5].
    """
    n = cipher.input_bits
    delta_bits = int_to_bits(delta, n)
    x0 = rng.integers(0, 2, size=(samples, n), dtype=np.uint8)                # Def. 2: X_i uniform
    keys = rng.integers(0, 2, size=(samples, cipher.key_bits), dtype=np.uint8)  # Def. 2: one K_i per sample
    x1 = x0 ^ delta_bits                                                     # X_i xor delta (A-021)
    for arr in (x0, x1, keys):
        arr.setflags(write=False)
    c0 = _encrypt_checked(cipher, x0, keys, rounds)
    c1 = _encrypt_checked(cipher, x1, keys, rounds)
    ones = np.count_nonzero(c0 ^ c1, axis=0)                                 # sum_i (E(X_i) xor E(X_i xor delta))_j
    freq = ones / samples                                                    # divide by t
    return float(np.mean(np.abs(_BIAS_CENTRE - freq)))                       # (1/n) sum_j |0.5 - .|


def _evaluate(scorer: Scorer, cipher: Cipher, delta: int, rounds: int, samples: int,
              rng: np.random.Generator) -> float:
    value = scorer(cipher, delta, rounds, samples, rng)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"scorer returned non-numeric value {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"scorer returned non-finite value {out}")
    return out


# --------------------------------------------------------------------------
# Algorithm 1 building blocks
# --------------------------------------------------------------------------
def _random_differences(rng: np.random.Generator, n: int, count: int) -> list[int]:
    """L1: RandomInt(0, 2^n - 1), ``count`` times, inclusive bounds (A-003)."""
    nbytes = (n + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE
    raw = rng.integers(0, _BYTE_VALUES, size=(count, nbytes), dtype=np.uint8)
    mask = (1 << n) - 1
    return [int.from_bytes(row.tobytes(), "big") & mask for row in raw]


def generate_offspring(population: Sequence[int], n: int, mutation_probability: float,
                       rng: np.random.Generator) -> list[tuple[int, int, int]]:
    """Alg. 1 L5-L15. Returns (i, j, child) for every pair i < j, in loop order.

    child = pop_i xor pop_j xor (m << p), with m = 1 iff u < M, u ~ U[0, 1),
    and p uniform in {0, ..., n-1}. The shift operator is read from a lost
    glyph (A-001). All u are drawn first, then all p (A-003).
    """
    n = _check_int("n", n, 1)
    size = len(population)
    pairs = [(i, j) for i in range(size) for j in range(i + 1, size)]  # L6-L7, exclusive bounds (A-002)
    u = rng.random(len(pairs))                                        # L8: RandomFloat(0, 1), [0, 1)
    positions = rng.integers(0, n, size=len(pairs))                   # L13: RandomInt(0, n - 1), inclusive
    out: list[tuple[int, int, int]] = []
    for k, (i, j) in enumerate(pairs):
        m = 1 if u[k] < mutation_probability else 0                   # L8-L12
        child = population[i] ^ population[j] ^ (m << int(positions[k]))  # L13
        out.append((i, j, child))
    return out


def _hygiene(values: Sequence[int], deduplicate: bool, exclude: Sequence[int] = ()) -> tuple[list[int], int, int]:
    """Drop zeros (A-004) and, optionally, repeats (A-005). Returns (kept, zeros, duplicates)."""
    seen = set(exclude)
    kept: list[int] = []
    zeros = duplicates = 0
    for value in values:
        if value == 0:
            zeros += 1
            continue
        if deduplicate:
            if value in seen:
                duplicates += 1
                continue
            seen.add(value)
        kept.append(value)
    return kept, zeros, duplicates


def _sort_order(scores: Sequence[float]) -> list[int]:
    """Indices sorting scores descending; stable, so ties keep input order (A-019)."""
    if len(scores) == 0:
        return []
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable").tolist()


def _sorted_desc(values: Sequence[int], scores: Sequence[float]) -> tuple[list[int], list[float]]:
    order = _sort_order(scores)
    return [values[i] for i in order], [scores[i] for i in order]


@dataclass(frozen=True)
class RoundResult:
    """Outcome of one run of Algorithm 1 at a fixed round count (SPECIFICATION §3)."""

    rounds: int
    differences: tuple[int, ...]               # Delta_r (A-007)
    scores: tuple[float, ...]                  # scores from the sort that selected Delta_r
    final_candidates: tuple[int, ...]          # full final sorted list (L16, L19)
    final_candidate_scores: tuple[float, ...]
    generations_completed: int
    evaluations: int
    candidates_generated: int                  # before hygiene, including the L1 draws
    zeros_discarded: int
    duplicates_discarded: int
    population_sizes: tuple[int, ...]          # after L3, then after each L17
    best_score_per_generation: tuple[Optional[float], ...]
    stopped_early: bool                        # A-023

    def as_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def _run_round(cipher: Cipher, rounds: int, cfg: _Config, seed: int, scorer: Scorer) -> RoundResult:
    """One run of Algorithm 1 (L1-L19) at ``rounds`` rounds."""
    n = cipher.input_bits
    keep = cfg.population_size
    search_rng = _generator(seed, (_STREAM_ROUND, rounds, _SUB_SEARCH))
    score_rng = _generator(seed, (_STREAM_ROUND, rounds, _SUB_SCORE))

    def score_all(values: Sequence[int]) -> list[float]:
        return [_evaluate(scorer, cipher, v, rounds, cfg.samples, score_rng) for v in values]

    # L1, with hygiene (A-004, A-005, A-024)
    raw = _random_differences(search_rng, n, cfg.initial_population)
    generated = len(raw)
    initial, zeros, duplicates = _hygiene(raw, cfg.deduplicate)
    initial_scores = score_all(initial)
    evaluations = len(initial)
    # L2-L3
    final_c, final_s = _sorted_desc(initial, initial_scores)
    population, pop_scores = final_c[:keep], final_s[:keep]
    sizes = [len(population)]
    best: list[Optional[float]] = [pop_scores[0] if pop_scores else None]
    completed = 0
    stopped_early = False

    for _ in range(cfg.generations):                       # L4 (A-002)
        if len(population) < 2:                            # A-023
            stopped_early = True
            break
        triples = generate_offspring(population, n, cfg.mutation_probability, search_rng)  # L5-L15
        generated += len(triples)
        exclude = population if (cfg.elitism and cfg.deduplicate) else ()
        children, z, d = _hygiene([c for _, _, c in triples], cfg.deduplicate, exclude)
        zeros += z
        duplicates += d
        child_scores = score_all(children)
        evaluations += len(children)
        pool = children + (list(population) if cfg.elitism else [])          # L17 / A-006
        pool_scores = child_scores + (list(pop_scores) if cfg.elitism else [])
        if not pool:                                       # A-023: keep the last selection
            stopped_early = True
            break
        final_c, final_s = _sorted_desc(pool, pool_scores)  # L16
        population, pop_scores = final_c[:keep], final_s[:keep]  # L17
        completed += 1
        sizes.append(len(population))
        best.append(pop_scores[0])

    return RoundResult(
        rounds=rounds,
        differences=tuple(population),
        scores=tuple(pop_scores),
        final_candidates=tuple(final_c),
        final_candidate_scores=tuple(final_s),
        generations_completed=completed,
        evaluations=evaluations,
        candidates_generated=generated,
        zeros_discarded=zeros,
        duplicates_discarded=duplicates,
        population_sizes=tuple(sizes),
        best_score_per_generation=tuple(best),
        stopped_early=stopped_early,
    )


def _paper_parameter_report(values: dict, names: Sequence[str]) -> dict:
    params: dict = {}
    sources: dict = {}
    for name in names:
        value = values[name]
        if name in PAPER_DEFAULTS and value == PAPER_DEFAULTS[name]:
            source = _PAPER_SOURCES[name]
        elif name in PAPER_REDUCED and value == PAPER_REDUCED[name]:
            source = _REDUCED_SOURCE
        elif name in _EXTRA_PAPER_VALUES and value == _EXTRA_PAPER_VALUES[name]:
            source = _PAPER_SOURCES[name]
        else:
            continue
        params[name] = value
        sources[name] = source
    return {"paper_parameters": params, "paper_parameter_sources": sources}


def _scorer_name(scorer: Optional[Scorer]) -> str:
    if scorer is None:
        return "bias_score (Definition 2)"
    return getattr(scorer, "__qualname__", type(scorer).__qualname__)


# --------------------------------------------------------------------------
# Public entry point (brief §7)
# --------------------------------------------------------------------------
def search(
    cipher: Cipher,
    *,
    seed: int,
    rounds: Optional[int] = None,
    max_rounds: Optional[int] = None,       # NOT IN PAPER: default cipher.nominal_rounds, see AMBIGUITY_LOG.md#A-011
    initial_population: int = PAPER_DEFAULTS["initial_population"],
    population_size: int = PAPER_DEFAULTS["population_size"],
    generations: int = PAPER_DEFAULTS["generations"],
    mutation_probability: float = PAPER_DEFAULTS["mutation_probability"],
    samples: int = PAPER_DEFAULTS["samples"],
    threshold: float = PAPER_DEFAULTS["threshold"],
    round_weighting: str = "linear",        # NOT IN PAPER: chosen as w_r = r, see AMBIGUITY_LOG.md#A-017
    deduplicate: bool = True,               # NOT IN PAPER: chosen as True, see AMBIGUITY_LOG.md#A-005
    elitism: bool = False,                  # Alg. 1 L17 (A-006)
    scorer: Optional[Scorer] = None,        # A-028: test hook; default is Definition 2
) -> dict:
    """Run the evolutionary optimizer.

    rounds=None runs the full §5.2 procedure: rounds 1..R+1, union, rescoring,
    weighted score. rounds=r runs Algorithm 1 once at r rounds.
    See SPECIFICATION.md §§3-4, 7.
    """
    validate_cipher(cipher)
    seed = _check_int("seed", seed, 0)
    if scorer is not None and not callable(scorer):
        raise TypeError("scorer must be callable or None")
    cfg = _build_config(
        initial_population=initial_population,
        population_size=population_size,
        generations=generations,
        mutation_probability=mutation_probability,
        samples=samples,
        threshold=threshold,
        round_weighting=round_weighting,
        deduplicate=deduplicate,
        elitism=elitism,
    )
    scorer_fn: Scorer = bias_score if scorer is None else scorer
    values = {**asdict(cfg), "starting_round": STARTING_ROUND}

    # ---- single-round mode: Algorithm 1 alone ----
    if rounds is not None:
        rounds = _check_int("rounds", rounds, 1)
        res = _run_round(cipher, rounds, cfg, seed, scorer_fn)
        names = [k for k in values if k not in ("threshold", "round_weighting", "starting_round")]
        out = {
            "mode": "single_round",
            "input_differences": list(res.differences),
            "score_per_difference": list(res.scores),
            "generations_completed": res.generations_completed,
            "evaluations": res.evaluations,
            "params_used": {
                **asdict(cfg),
                "rounds": rounds,
                "max_rounds": None,
                "scorer": _scorer_name(scorer),
                "mode": "single_round",
            },
            "seed": seed,
            **_paper_parameter_report(values, names),
            "terminated_by": "single_round",
            "per_round": [res.as_dict()],
        }
        if scorer is None:
            out["encryptions"] = 2 * cfg.samples * res.evaluations
        return out

    # ---- full procedure (§5.2, "Accounting for the Starting Round") ----
    cap = cipher.nominal_rounds if max_rounds is None else max_rounds
    if cap is None:
        raise ValueError("max_rounds is required when cipher.nominal_rounds is None (A-011)")
    cap = _check_int("max_rounds", cap, STARTING_ROUND)

    per_round: list[tuple[RoundResult, bool]] = []
    kept: list[RoundResult] = []
    terminated_by = "round_cap"
    for r in range(STARTING_ROUND, cap + 1):                          # O1
        res = _run_round(cipher, r, cfg, seed, scorer_fn)
        passed = any(s > cfg.threshold for s in res.scores)            # O2: strict > (A-009)
        per_round.append((res, passed))
        if not passed:
            terminated_by = "threshold"
            break
        kept.append(res)                                               # all members kept (A-010)
    big_r = len(kept)

    # O4: set union, ordered by first appearance (A-018)
    union: list[int] = []
    seen: set[int] = set()
    for res in kept:
        for d in res.differences:
            if d not in seen:
                seen.add(d)
                union.append(d)

    # O5: rescore every member at every round 1..R with fresh samples (A-016)
    rescore_rng = _generator(seed, (_STREAM_RESCORE,))
    matrix = [
        [_evaluate(scorer_fn, cipher, d, r, cfg.samples, rescore_rng)
         for r in range(STARTING_ROUND, big_r + 1)]
        for d in union
    ]
    rescore_evaluations = len(union) * big_r

    # O6: uniform and round-weighted sums (A-017)
    uniform = [math.fsum(row) for row in matrix]
    linear = [math.fsum(r * b for r, b in enumerate(row, start=STARTING_ROUND)) for row in matrix]
    selected = linear if cfg.round_weighting == "linear" else uniform

    # O7: rank by the selected score (stable)
    order = _sort_order(selected)
    evaluations = sum(res.evaluations for res, _ in per_round) + rescore_evaluations
    out = {
        "mode": "full",
        "input_differences": [union[i] for i in order],
        "score_per_difference": [selected[i] for i in order],
        "generations_completed": sum(res.generations_completed for res, _ in per_round),
        "evaluations": evaluations,
        "params_used": {
            **asdict(cfg),
            "starting_round": STARTING_ROUND,
            "rounds": None,
            "max_rounds": cap,
            "scorer": _scorer_name(scorer),
            "mode": "full",
        },
        "seed": seed,
        **_paper_parameter_report(values, list(values)),
        "rounds_reached": big_r,
        "terminated_by": terminated_by,
        "per_round": [{**res.as_dict(), "passed_threshold": passed} for res, passed in per_round],
        "rescored_matrix": [matrix[i] for i in order],
        "score_uniform": [uniform[i] for i in order],
        "score_round_weighted": [linear[i] for i in order],
    }
    if scorer is None:
        out["encryptions"] = 2 * cfg.samples * evaluations
    return out
