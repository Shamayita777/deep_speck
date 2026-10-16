# II-2 Clean-Room Brief — independent reimplementation of AutoND's Algorithm 1

**Audience: the implementer.** You are being asked to implement one
algorithm from a published paper, from the paper alone, without looking
at the authors' released code. Read this whole document before writing
anything.

This brief is itself an audit artifact. It is committed and hash-bound so
that a reviewer can verify exactly what you were given and what you were
denied.

---

## 1. Why you specifically

This is stage II-2 of a five-stage implementation-integrity audit. Its
entire scientific value rests on one property: **the implementation was
derived from the paper's description, not from the authors' source.**

If that property fails, II-2 is worthless and so is II-4, which compares
your implementation against the authors' as two arms of a controlled
experiment. An implementation that has absorbed the original's design
cannot disagree with it in an informative way.

**The assistant that wrote this brief is disqualified from implementing
it.** It has read the released `optimizer.py`, including the internals of
its search function. Anything it wrote would be a paraphrase of code it
has seen, and a reviewer would be right to discount it. That is the only
reason this is being delegated, and it is stated here so the limitation
is on the record rather than implied.

You are being asked to do the thing it cannot.

---

## 2. What you must implement

**AutoND's Algorithm 1**, as described in:

> *A Deeper Look at Machine Learning-Based Cryptanalysis*
> ePrint 2022/1467 — use the version served at the canonical ePrint URL
> (the last of four revisions, 2024-01-24). The audit's paper-to-source
> comparison is bound to that revision.

Work from **Algorithm 1 and its surrounding prose**. Where the paper
gives notation, use the paper's notation. Where the paper's text and its
pseudocode appear to differ, implement what you judge the paper to
specify and **record the discrepancy** (§6).

Nothing in this brief paraphrases the algorithm, on purpose. Describing
it here would substitute this document's reading of the paper for yours,
which is precisely the independence being purchased. Read the paper.

---

## 3. Hard prohibitions

These are not preferences. Violating any of them invalidates the stage,
and you must say so immediately rather than continue.

**You must NOT, at any point:**

1. Open, clone, browse, download, search, or read **any part of the
   AutoND repository** — not the source, not the README, not the issues,
   not the commit history, not a fork, not a mirror, not a vendored copy
   inside someone else's project.
2. Read any **other** implementation of this algorithm: a reimplementation,
   a blog post containing its code, a notebook, a thesis appendix, a
   StackOverflow answer quoting it, an LLM-generated version.
3. Search the web for the algorithm's *implementation*. (Looking up
   standard background — what a differential is, how a genetic algorithm
   works, a NumPy API — is fine and expected.)
4. Ask anyone, human or model, what the released code does.
5. Accept any description of the released code if someone offers one.
   Stop and report it instead.

**You will not be given,** and must not seek:

6. **The expected results.** You are not told which input difference the
   search is supposed to return, how many it returns, or what round it is
   supposed to reach. Knowing the target would let you tune toward it,
   consciously or not. The audit holds those values and will compare
   afterwards.
7. The prior stages' findings, or the conversation that produced this
   brief.

If you have **already** read the AutoND source at some earlier point, say
so now, before starting. That is not a failure; it is a fact the audit
needs. An implementation by someone with prior exposure is still useful,
but it must be labelled honestly, and only you can report it.

---

## 4. What you may use

- The paper (PDF), including its algorithms, figures, tables and prose.
- General knowledge of symmetric cryptanalysis, differential
  cryptanalysis, evolutionary/genetic algorithms, and machine learning.
- Standard library and ordinary scientific Python: `numpy`, and `pytest`
  or `unittest`. Ask before adding any other dependency.
- Published work *other than* this paper for background concepts —
  cited if it influenced a design choice.

Target environment, because the audit runs everything under one pinned
stack: **Python 3.11, NumPy 1.26.4.** Do not use NumPy 2.x APIs or rely
on NEP 50 promotion behaviour.

---

## 5. Deliverables

Create a package at:

```
audit/targets/autond/ii2_independent/
    __init__.py
    algorithm1_independent.py     the implementation
    cipher_interface.py           how a cipher is supplied to it (§7)
    SPECIFICATION.md              your reading of the paper (§6)
    AMBIGUITY_LOG.md              every underspecification (§6)
    PROVENANCE.md                 your declaration (§8)
    tests/
        __init__.py
        test_algorithm1_independent.py
```

Mirror this layout; the audit's II-4 harness expects the package path.
Within the module, name things however your reading of the paper
suggests.

**Code requirements**

- Deterministic given a seed. Expose the seed; never read a global RNG.
  The audit must be able to rerun you and get identical output.
- No network access at runtime. No file downloads.
- No training of neural networks in this module unless the paper's
  Algorithm 1 itself requires it. If it does, the scoring function must
  be injectable so the audit can run the search without a GPU.
- Pure functions where possible; no module-level mutable state.
- Every magic number traceable to a paper location, in a comment:
  `# paper §4.2` or `# Algorithm 1, line 7`. If a constant is **your**
  choice because the paper does not give one, mark it
  `# NOT IN PAPER: chosen as ... see AMBIGUITY_LOG.md#A-00n`.

**Tests**

Test the properties the paper states, not the results the audit expects
(which you do not have). Good targets: determinism under a fixed seed;
population invariants the paper asserts; termination; input validation;
behaviour at stated boundaries. Avoid asserting any specific difference
or round number — if you find yourself wanting to, that number is not
something you should know.

---

## 6. The ambiguity log is a primary deliverable

Treat this as equal in importance to the code.

The paper will not fully determine the implementation. An earlier stage
of this audit independently found **8 architectural properties the paper
leaves underspecified** (that count is for a different component; you are
not told which properties, to avoid steering you). Expect the same
experience with Algorithm 1.

When the paper does not determine something, **do not guess what the
authors probably did.** That instinct is the leak this whole exercise is
built to prevent. Instead:

1. Record it in `AMBIGUITY_LOG.md` with an id (`A-001`, …).
2. State what the paper does and does not say, with the location.
3. State the choice you made and why.
4. State what a different reasonable choice would have been, and whether
   you expect it to change results.

`SPECIFICATION.md` is your prose reading of the algorithm before and
separate from the code: inputs, outputs, each step, the termination
condition, the scoring, in your own words with paper citations. Write it
**first**, then implement from your own specification. If the code and
the specification diverge later, fix one of them and note why.

The ambiguity log is scientifically valuable in its own right: it is a
second, independent measurement of how reproducible the paper is from its
text alone. A log with twenty honest entries is a better result than a log
with two and a confident implementation.

---

## 7. Interface contract

II-4 compares your implementation against the authors' as two arms, so
the audit must be able to call both through one adapter.

**This contract was written from the audit's needs and from the paper's
own description of the algorithm's inputs and outputs. It is deliberately
minimal and does not mirror the released code's function signature** —
mirroring it would be a leak of exactly the kind this brief forbids. If
any part of it looks like it is forcing an internal design on you, say
so: the contract will be changed rather than your implementation bent to
fit it.

Expose one callable:

```python
def search(cipher, *, seed: int, **params) -> dict:
    """
    Run Algorithm 1 for one cipher and return its outcome.

    Returns a dict containing at least:
      "input_differences"        the difference(s) the search returns,
                                 as a list of ints
      "score_per_difference"     the score your scoring step assigned to
                                 each returned difference, same order
      "generations_completed"    int
      "evaluations"              how many scoring calls were made
      "params_used"              every parameter actually in effect,
                                 including defaults you chose
      "seed"                     the seed used
      "paper_parameters"         {name: value} for each parameter whose
                                 value you took from the paper, with the
                                 paper location as the value's comment or
                                 a parallel "paper_parameter_sources" map
    """
```

Add any other keys your reading produces; the audit reads what is there
and ignores what it does not need. If the paper's algorithm does not
produce one of the keys above, omit it and explain in
`AMBIGUITY_LOG.md` — do not invent a value to satisfy this contract.

`cipher_interface.py` defines what a `cipher` argument is: whatever your
reading of the paper requires the algorithm to know about a cipher (for
instance a way to encrypt, and the parameters the paper says the search
needs). Define it in your own terms and document it; the audit will write
the adapter that supplies real ciphers to it.

---

## 8. Provenance declaration

`PROVENANCE.md` must state, in your own words:

- Which paper revision you worked from, and its sha256 if you hashed it.
- That you did not read the AutoND repository or any other
  implementation, **or** exactly what you did read, if anything.
- Any prior exposure to this algorithm's released implementation,
  including before this task.
- Every external source you consulted, and what it influenced.
- Any point at which you were tempted to look up the original, and what
  you did instead. (This is not a confession; it marks where the paper is
  weakest, which is useful.)
- Your model/version or name, the date, and how long it took.

An implementation without this file cannot be used, because the audit
cannot characterise it.

---

## 9. What happens next, so you know what matters

After you deliver:

- The audit verifies your package runs deterministically and matches your
  own specification.
- II-4 runs your implementation and the authors' on the same ciphers and
  compares outcomes under a preregistered statistical rule, with a margin
  fixed in advance.
- **Disagreement is a legitimate, publishable result.** You are not being
  asked to reproduce the authors' numbers, and you will not be judged on
  whether you do. An honest implementation that disagrees is far more
  valuable than one nudged toward agreement.
- Your ambiguity log is reported as a finding about the paper.

If at any point you cannot proceed without information you are forbidden
to look up, **stop and say so**, naming what you need. That outcome —
"Algorithm 1 is not implementable from the paper alone without assumption
X" — is a finding, and reporting it is a success, not a failure.

---

## 10. Checklist before you deliver

- [ ] `SPECIFICATION.md` written before the code, with paper citations
- [ ] every non-paper constant marked `# NOT IN PAPER` and logged
- [ ] `AMBIGUITY_LOG.md` complete, with alternatives considered
- [ ] deterministic under a fixed seed; verified by a test
- [ ] no network, no downloads, no global RNG, no module-level state
- [ ] runs under Python 3.11 / NumPy 1.26.4
- [ ] tests assert paper-stated properties, not expected results
- [ ] `PROVENANCE.md` complete and honest, including any prior exposure
- [ ] you did not look at the AutoND repository
