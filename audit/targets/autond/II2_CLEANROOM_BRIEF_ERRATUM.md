# Erratum to II2_CLEANROOM_BRIEF.md

The brief (sha256 4bd8ee37b566be1d8a52a2f43728e297dac9026ec36cb256bf477891a20d985a)
is NOT edited. Corrections are recorded here.

E-1. Wrong paper title. §2 of the brief names "A Deeper Look at
Machine Learning-Based Cryptanalysis". That is Benamira, Gerault,
Peyrin and Tan, ePrint 2021/287 - a different paper, cited inside the
target paper. The correct title for ePrint 2022/1467 is "A
Cipher-Agnostic Neural Training Pipeline with Automated Finding of
Good Input Differences" (Bellini, Gerault, Hambitzer, Rossi). The
ePrint number and revision given in the brief were correct, so the
intended paper was never in doubt and II-3's paper-to-source
comparison - which binds the ePrint number, not the title - is
unaffected.

Found by the II-2 implementer, unprompted, and reported as
AMBIGUITY_LOG S-1 before writing any code. Verified against
eprint.iacr.org/2022/1467.

E-2. Algorithm number possibly wrong. The brief says "Algorithm 1".
One independent extraction of the PDF reports the evolutionary
optimizer as Algorithm 2, with Algorithm 1 being Gohr's greedy
optimizer; the implementer's extraction reports the reverse. To be
settled against the rendered PDF. The implementer read the
evolutionary optimizer in either case.

E-3. Incomplete blinding, unavoidable. Brief §3(6) says the
implementer is not told the expected results. The paper publishes its
own returned differences and round counts (Tables 3 and 6, §5.1, §7),
so an implementer reading the only permitted source necessarily sees
them. Disclosed by the implementer as AMBIGUITY_LOG S-3. The blinding
that II-4 depends on is intact: the comparison is against what the
released code returns, which the implementer has not seen.
