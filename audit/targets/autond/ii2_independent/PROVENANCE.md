# II-2 Provenance Declaration

## Implementer
- **Model:** a large language model, running in a chat interface with no file
  access and no code execution.
- **Model identity withheld from this file.** The specific model and version
  are known to the audit and are deliberately not recorded here. They are
  available to reviewers on request. This is stated rather than omitted so
  that a reader knows the information exists and is obtainable; the
  independence caveat below does not depend on it, because it is stated
  conservatively: corpus-level independence is NOT claimed either way.
- **Date:** 2026-10-16.
- **Duration:** a small number of conversational turns in one session. I have
  no clock and cannot report elapsed time.

## Paper worked from
- The PDF attached to the task: ePrint 2022/1467, "A Cipher-Agnostic Neural
  Training Pipeline with Automated Finding of Good Input Differences" (Bellini,
  Gerault, Hambitzer, Rossi). The title was confirmed by the auditor.
- **Identifying features:**
  - last printed page number 29;
  - 2 figures and 9 tables;
  - Algorithm 1 is the evolutionary optimizer (§5.2);
  - Algorithm 2 is Gohr's optimizer (Appendix A).
- **Revision unknown.** Whether the given PDF is the bound revision is
  unresolved. A claimed content difference was withdrawn (A-031). The
  operator-verified §5.2 text matches the given PDF.
- **No sha256.** This session had no code execution, so I could not hash the
  file.
- **How I read it.** I worked from a text rendering in which some math glyphs
  were lost (AMBIGUITY_LOG S-2).

## What I did not read
- I did not open, search for, browse, or read any part of the AutoND
  repository: not the source, README, issues, history, forks or mirrors.
- I did not read any other implementation of this algorithm.
- I made no tool calls of any kind in this session: no web search, no page
  fetches.
- I did not follow the GitHub URL that the paper prints.

## Prior exposure
- My training data includes a large amount of public web and code-hosting
  content. The AutoND repository is public, so I cannot rule out that its
  source was in my training data. I cannot inspect my training data to check.
- I am not knowingly drawing on any recollection of it. Every design choice is
  either cited to the paper or logged in AMBIGUITY_LOG.md with my reasoning.
- The implementation was produced by a large language model working from the
  paper alone, with no access to the released repository during the session.
  The model's training data is not inspectable and may include the public
  repository; it may also share a training corpus with the model that
  performed the paper-to-source comparison. II-2 is therefore independent at
  the session level but not demonstrably independent at the corpus level, and
  II-4 should interpret agreement between arms accordingly.
- **Suggested label:** "clean-room in-session; prior exposure via training
  corpus not excludable."
- If II-4 needs stronger independence, a human implementer or a model from a
  different family would be a useful additional arm.

## Information received beyond the paper
1. Confirmation of the paper's title and ePrint number.
2. The canonical revision's numbering (A-031), including that its Gohr
   optimizer uses a shift (A-001).
3. A sentence attributed to the canonical revision specifying w_r = r. I
   adopted it on attribution, because I could not find it in my PDF, and
   flagged it for verification. The auditor later retracted it: an automated
   extraction had invented it. It is removed from code and documents.
   The canonical-numbering report in item 2 came from the same kind of
   extraction and is now treated as unverified.
4. ARCH-09, a prior-stage finding about the DBitNet figure. Brief §3(7) says
   such findings are withheld. It is unrelated to this package and nothing
   here depends on it.

5. The operator's direct verification of the §5.2 text and a rendered image
   of the S_δ equation. This settled A-017 and refuted S-2 item 4.
6. Auditor-measured sensitivity of the ranking to the weighting (A-017).

None of these described the released code.

## Things in the paper the brief did not anticipate
- The paper publishes its own returned differences and round counts (Tables 3
  and 6, §5.1, §7). I read them because the paper is a permitted source. I did
  not use them to make any choice, and no test asserts them.
- §8 contains one sentence describing the repository's reduced parameter set.
  I reported it (S-4) and did not act on it. The `PAPER_REDUCED` preset is
  taken from §5.4's text only.

## External knowledge used
All from memory; nothing was looked up.
- Standard genetic-algorithm and differential-cryptanalysis concepts.
- The normal approximation to the binomial and half-normal moments, used to
  check the §5.4 baseline. This supported A-013 and A-014 and test 7.
- The SplitMix64 finalizer (Steele, Lea and Flood, 2014), used only as a test
  cipher's pseudo-random function.
- NumPy 1.26 APIs: `Generator`, `SeedSequence(spawn_key=...)`, `packbits`,
  `unpackbits`, stable `argsort`.

## Where the paper is weakest
These are the points where I would most have wanted to consult the original.
In each case I logged the question instead.
- **Final-score weighting (A-017).** The formula and the prose disagree. This
  was later confirmed by visual inspection to be a contradiction in the paper
  itself, not an extraction artefact.
- **Zero and duplicate differences (A-004, A-005).** Nothing in the paper says
  how to stop δ = 0 or repeated copies from taking over the population, and
  either choice changes the search dynamics.
- **The lost operator in L13 (A-001).**
- **What Algorithm 1 returns (A-007).** L19 and the prose give different sizes.

## Verification status
- **Not executed by me.** No test or line of code was run in my session.
- **First execution, by the audit:** 43/43 tests passed on Python 3.11.17 /
  NumPy 1.26.4 with no modifications, deterministic across separate processes,
  and seven additional paths probed.
- **Re-run after the A-017 attribution change:** 43/43 pass. The added
  assertion (`"round_weighting" not in res["paper_parameters"]`) sits inside
  an existing test, so the count is unchanged. Verified across both
  weightings and both modes that the weighting appears in neither
  `paper_parameters` nor `paper_parameter_sources`, while `params_used`
  still records it.
- **Operator-side execution:** 43/43 pass on the operator's machine under
  the pinned audit environment.
- **Cost of a full run.** At paper defaults, each round costs about 25,800
  scorer calls of 2·10⁴ encryptions each. This is feasible only for ciphers
  with a vectorized `encrypt`.
