"""Cryptographic known-answer and implementation-agreement tests for the
published Algorithms 1 and 2 (P1 p. 32). They establish that the code computes
what docs/CRYPTO_CORRECTNESS.md says; they are not evidence that Table 7 is
reproduced (that is the gate)."""

import numpy as np
import pytest

from audit.cryptography.experiments.ce5 import akr as A
from audit.cryptography.experiments.ce5 import reference as REF
from audit.cryptography.experiments.ce5 import verify as V
from audit.cryptography.gohr import speck as S

DIFF = (0x0040, 0x0000)


@pytest.fixture(scope="module")
def toy4():
    return A.build_table(200_000, 4, DIFF, rng=np.random.default_rng(1)), 200_000


def _u(v):
    return np.array([v], dtype=np.uint16)


def test_reference_speck_matches_published_test_vector():
    assert REF.self_test()


def test_gohr_speck_matches_published_test_vector():
    key = [_u(w) for w in REF.TEST_VECTOR["key"]]
    ct = S.encrypt((_u(0x6574), _u(0x694C)), S.expand_key(key, 22))
    assert (int(ct[0][0]), int(ct[1][0])) == REF.TEST_VECTOR["ct"]


def test_key_schedules_agree_on_random_keys():
    rng = np.random.default_rng(2)
    keys = rng.integers(0, 1 << 16, size=(4, 50), dtype=np.uint16)
    ks = S.expand_key(keys, 5)
    for j in range(50):
        assert [int(k[j]) for k in ks] == REF.key_schedule(tuple(int(w) for w in keys[:, j]), 5)


def test_encryption_agrees_with_reference_at_five_rounds():
    rng = np.random.default_rng(3)
    keys = rng.integers(0, 1 << 16, size=(4, 50), dtype=np.uint16)
    pl, pr = (rng.integers(0, 1 << 16, size=50, dtype=np.uint16) for _ in range(2))
    cl, cr = S.encrypt((pl, pr), S.expand_key(keys, 5))
    for j in range(50):
        assert (int(cl[j]), int(cr[j])) == REF.encrypt((int(pl[j]), int(pr[j])),
                                                         tuple(int(w) for w in keys[:, j]), 5)


def test_decryption_inverts_encryption_in_both_implementations():
    rng = np.random.default_rng(4)
    for _ in range(200):
        x, y, k = (int(v) for v in rng.integers(0, 1 << 16, size=3))
        assert REF.round_dec(*REF.round_enc(x, y, k), k) == (x, y)
        c = S.enc_one_round((_u(x), _u(y)), _u(k))
        assert tuple(int(v[0]) for v in S.dec_one_round(c, _u(k))) == (x, y)


def test_right_word_after_decryption_is_key_free_as_stated_on_p17():
    rng = np.random.default_rng(5)
    for _ in range(100):
        l, r = (int(v) for v in rng.integers(0, 1 << 16, size=2))
        rights = {REF.round_dec(l, r, k)[1] for k in rng.integers(0, 1 << 16, size=20)}
        assert rights == {REF.rotr(l ^ r, 2)}


def test_word_order_and_bit_encoding():
    rng = np.random.default_rng(6)
    w = [rng.integers(0, 1 << 16, size=64, dtype=np.uint16) for _ in range(4)]
    X = S.convert_to_binary(w)
    assert X[0, 0] == (int(w[0][0]) >> 15) & 1 and X[0, 16] == (int(w[1][0]) >> 15) & 1
    assert all(np.array_equal(a, b) for a, b in zip(w, A.words_from_binary(X)))
    assert np.array_equal(V._bits(w), X)


def test_delta_v_is_the_xor_of_the_two_v_values_not_delta_r():
    l0, r0, l1, r1 = 0x1234, 0xABCD, 0x0F0F, 0x00FF
    dl, dv = A.differences(*(_u(x) for x in (l0, r0, l1, r1)))
    assert int(dl[0]) == l0 ^ l1
    assert int(dv[0]) == (l0 ^ r0) ^ (l1 ^ r1)
    assert int(dv[0]) != r0 ^ r1
    assert REF.deltas((l0, r0), (l1, r1)) == (int(dl[0]), int(dv[0]))


def test_table_is_indexed_by_dl_dv_and_not_dl_dr():
    rng = np.random.default_rng(7)
    c = A.ciphertext_pairs(1, 4, DIFF, real=True, rng=rng)
    counts = A.build_table(1, 4, DIFF, rng=np.random.default_rng(7))
    cell = int(np.flatnonzero(counts)[0])
    l0, r0, l1, r1 = (int(x[0]) for x in c)
    assert cell == REF.cell(l0 ^ l1, l0 ^ r0 ^ l1 ^ r1, A.MASK)
    if (r0 ^ r1) & 0xFF8F != (l0 ^ r0 ^ l1 ^ r1) & 0xFF8F:
        assert cell != REF.cell(l0 ^ l1, r0 ^ r1, A.MASK)


def test_mask_weight_cell_layout_and_size():
    idx = A.MaskedIndex()
    assert A.mask_weight() == 26 and idx.size == 1 << 26 and idx.shift == 13
    rng = np.random.default_rng(8)
    for dl, dv in rng.integers(0, 1 << 16, size=(300, 2)):
        assert int(idx(_u(dl), _u(dv))[0]) == REF.cell(int(dl), int(dv), A.MASK)


def test_cell_ignores_unmasked_bits_and_separates_masked_ones():
    idx = A.MaskedIndex()
    rng = np.random.default_rng(9)
    dl = rng.integers(0, 1 << 16, size=5000, dtype=np.uint16)
    dv = rng.integers(0, 1 << 16, size=5000, dtype=np.uint16)
    noise = np.uint16(~0xFF8F & 0xFFFF)
    assert np.array_equal(idx(dl, dv), idx(dl ^ noise, dv ^ noise))
    assert len(set(idx(dl, dv).tolist())) == len({(int(a) & 0xFF8F, int(b) & 0xFF8F) for a, b in zip(dl, dv)})


def test_candidate_loop_is_the_published_2_to_the_12():
    assert A.N_CANDIDATES == 4096 and np.array_equal(A.CANDIDATES, np.arange(4096))
    assert REF.N_CANDIDATES == 4096
    assert A.relevant_key_bits(A.MASK) == 12


def test_relevant_key_bits_derivation():
    assert A.relevant_key_bits((0xFFFF, 0xFFFF)) == 15
    assert A.relevant_key_bits((0x0F80, 0x0000)) == 4
    assert A.relevant_key_bits((0x1F80, 0x0003)) == 10


def test_masked_decryption_independent_of_key_bits_12_to_15():
    rng = np.random.default_rng(10)
    for _ in range(300):
        l0, r0, l1, r1, k = (int(v) for v in rng.integers(0, 1 << 16, size=5))
        base = REF.deltas(REF.round_dec(l0, r0, k), REF.round_dec(l1, r1, k))
        for hi in range(16):
            kk = (k & 0x0FFF) | (hi << 12)
            alt = REF.deltas(REF.round_dec(l0, r0, kk), REF.round_dec(l1, r1, kk))
            assert base[0] & 0xFF8F == alt[0] & 0xFF8F and base[1] & 0xFF8F == alt[1] & 0xFF8F


def test_key_bit_11_does_change_the_masked_cell_somewhere():
    rng = np.random.default_rng(11)
    changed = 0
    for _ in range(300):
        l0, r0, l1, r1, k = (int(v) for v in rng.integers(0, 1 << 16, size=5))
        a = REF.deltas(REF.round_dec(l0, r0, k), REF.round_dec(l1, r1, k))
        b = REF.deltas(REF.round_dec(l0, r0, k ^ 0x0800), REF.round_dec(l1, r1, k ^ 0x0800))
        changed += (a[0] & 0xFF8F, a[1] & 0xFF8F) != (b[0] & 0xFF8F, b[1] & 0xFF8F)
    assert changed > 0


def test_published_loop_equals_all_subkey_average_for_the_published_mask(toy4):
    counts, n = toy4
    rng = np.random.default_rng(12)
    for real in (True, False):
        c = A.ciphertext_pairs(30, 5, DIFF, real=real, rng=rng)
        assert np.array_equal(A.akr_scores(*c, counts, n), A.full_average(*c, counts, n))


def test_literal_2_to_the_12_differs_from_2_to_the_13_and_16_when_more_bits_matter():
    mask = (0xFFFF, 0x0000)
    idx = A.MaskedIndex(mask)
    counts = A.build_table(100_000, 4, DIFF, index=idx, rng=np.random.default_rng(13))
    c = A.ciphertext_pairs(20, 5, DIFF, real=True, rng=np.random.default_rng(14))
    s12 = A.akr_scores(*c, counts, 100_000, index=idx)
    s13 = A.akr_scores(*c, counts, 100_000, index=idx, candidates=np.arange(1 << 13))
    s16 = A.full_average(*c, counts, 100_000, index=idx)
    assert not np.array_equal(s12, s16) and not np.array_equal(s12, s13)
    for j in range(2):
        ref = REF.score((int(c[0][j]), int(c[1][j])), (int(c[2][j]), int(c[3][j])), counts, 100_000, mask)
        assert abs(ref - s12[j]) <= 1e-15


def test_vectorised_scores_match_independent_reference(toy4):
    counts, n = toy4
    rng = np.random.default_rng(15)
    for real in (True, False):
        c = A.ciphertext_pairs(4, 5, DIFF, real=real, rng=rng)
        fast = A.akr_scores(*c, counts, n)
        for j in range(4):
            ref = REF.score((int(c[0][j]), int(c[1][j])), (int(c[2][j]), int(c[3][j])), counts, n, A.MASK)
            assert abs(fast[j] - ref) <= 1e-15
            assert REF.decide(ref, A.MASK) == bool(A.akr_decide(fast[j:j + 1])[0])


def test_correct_subkey_recovers_the_four_round_cell(toy4):
    counts, n = toy4
    idx = A.MaskedIndex()
    rng = np.random.default_rng(16)
    keys = rng.integers(0, 1 << 16, size=(4, 8), dtype=np.uint16)
    p0l, p0r = (rng.integers(0, 1 << 16, size=8, dtype=np.uint16) for _ in range(2))
    ks = S.expand_key(keys, 5)
    c0 = S.encrypt((p0l, p0r), ks)
    c1 = S.encrypt((p0l ^ np.uint16(DIFF[0]), p0r), ks)
    x0 = S.encrypt((p0l, p0r), ks[:4])
    x1 = S.encrypt((p0l ^ np.uint16(DIFF[0]), p0r), ks[:4])
    for j in range(8):
        sl = slice(j, j + 1)
        got = A.akr_scores(c0[0][sl], c0[1][sl], c1[0][sl], c1[1][sl], counts, n, candidates=[ks[4][j]])
        dl, dv = A.differences(x0[0][sl], x0[1][sl], x1[0][sl], x1[1][sl])
        assert got[0] == counts[idx(dl, dv)][0] / n


def test_scalar_algorithm_1_matches_vectorised_counts_in_distribution(toy4):
    counts, n = toy4
    ref = REF.build_approx_ddt(4, 20_000, A.MASK, seed=17)
    top = np.argsort(counts)[::-1][:20]
    for i in top[:5]:
        p = counts[i] / n
        assert abs(ref.get(int(i), 0) / 20_000 - p) < 5 * np.sqrt(p * (1 - p) / 20_000) + 1e-3


def test_table_counts_every_pair_and_rejects_overflow(toy4):
    counts, n = toy4
    assert counts.dtype == np.uint32 and int(counts.sum(dtype=np.uint64)) == n
    with pytest.raises(ValueError):
        A.build_table(1 << 32, 4, DIFF)


def test_uint64_accumulation_and_normalisation():
    idx = A.MaskedIndex((0x0F80, 0x000F))
    c = A.ciphertext_pairs(5, 5, DIFF, real=True, rng=np.random.default_rng(18))
    full = np.full(idx.size, np.iinfo(np.uint32).max, dtype=np.uint32)
    assert np.all(A.akr_scores(*c, full, 1, index=idx) == float(np.iinfo(np.uint32).max))
    assert np.all(A.akr_scores(*c, np.full(idx.size, 7, dtype=np.uint32), 7, index=idx) == 1.0)


def test_unseen_differences_score_zero_and_decide_random():
    counts = np.zeros(1 << 26, dtype=np.uint32)
    s = A.akr_scores(*A.ciphertext_pairs(4, 5, DIFF, real=True, rng=np.random.default_rng(19)), counts, 10)
    assert np.all(s == 0) and not A.akr_decide(s).any()


def test_decision_threshold_is_strict_in_both_implementations():
    t = 2.0 ** -26
    assert list(A.akr_decide(np.array([t, np.nextafter(t, 1)]))) == [False, True]
    assert not REF.decide(t, A.MASK) and REF.decide(np.nextafter(t, 1), A.MASK)


def test_real_pairs_score_higher_than_random_on_toy_table(toy4):
    counts, n = toy4
    rng = np.random.default_rng(20)
    real = A.akr_scores(*A.ciphertext_pairs(300, 5, DIFF, real=True, rng=rng), counts, n)
    rand = A.akr_scores(*A.ciphertext_pairs(300, 5, DIFF, real=False, rng=rng), counts, n)
    assert np.mean(A.akr_decide(real)) > 0.8 and np.mean(~A.akr_decide(rand)) > 0.8


def test_independent_table_check_accepts_right_rounds_and_rejects_wrong_ones(toy4):
    counts, n = toy4
    assert REF.table_check(counts, n, rounds=4, diff=DIFF, mask=A.MASK, n_sample=20_000)["p_value"] >= 1e-3
    for r in (3, 5):
        assert REF.table_check(counts, n, rounds=r, diff=DIFF, mask=A.MASK, n_sample=20_000)["p_value"] < 1e-12
    assert REF.table_check(counts, n, rounds=4, diff=(0x0000, 0x0040), mask=A.MASK,
                           n_sample=20_000)["p_value"] < 1e-12


def test_independent_table_check_rejects_a_dl_dr_table():
    rng = np.random.default_rng(21)
    idx = A.MaskedIndex()
    c0l, c0r, c1l, c1r = A.ciphertext_pairs(200_000, 4, DIFF, real=True, rng=rng)
    wrong = np.zeros(idx.size, dtype=np.uint32)
    np.add.at(wrong, idx(c0l ^ c1l, c0r ^ c1r), 1)
    assert REF.table_check(wrong, 200_000, rounds=4, diff=DIFF, mask=A.MASK, n_sample=20_000)["p_value"] < 1e-12


def test_difference_score_reads_the_ciphertext_cell():
    counts = A.build_table(100_000, 5, DIFF, rng=np.random.default_rng(22))
    c = A.ciphertext_pairs(50, 5, DIFF, real=True, rng=np.random.default_rng(23))
    dl, dv = A.differences(*c)
    assert np.array_equal(A.difference_scores(*c, counts, 100_000), counts[A.MaskedIndex()(dl, dv)] / 100_000)


def _cells_all_keys(c, idx):
    k = np.arange(1 << 16, dtype=np.uint16)[None, :]
    c0l, c0r, c1l, c1r = (np.asarray(a, dtype=np.uint16) for a in c)
    r0 = S.ror(c0l ^ c0r, S.BETA())
    r1 = S.ror(c1l ^ c1r, S.BETA())
    dl = S.rol((c0l[:, None] ^ k) - r0[:, None], S.ALPHA()) ^ S.rol((c1l[:, None] ^ k) - r1[:, None], S.ALPHA())
    return idx(dl, dl ^ (r0 ^ r1)[:, None])


def test_exhaustive_over_all_subkeys_cell_depends_only_on_key_bits_0_to_11():
    idx = A.MaskedIndex()
    for real in (True, False):
        c = A.ciphertext_pairs(400, 5, DIFF, real=real, rng=np.random.default_rng(31 + real))
        cells = _cells_all_keys(c, idx)
        k = np.arange(1 << 16)
        assert np.array_equal(cells, cells[:, k & 0x0FFF])
        assert np.any(cells[:, k] != cells[:, k ^ 0x0800])


def test_masked_cell_representation_is_lossless_and_unaliased():
    v = np.arange(1 << 16, dtype=np.uint16)
    lut = A._compaction_lut(0xFF8F)
    assert np.array_equal(lut[v], lut[v & np.uint16(0xFF8F)])
    masked = np.unique(v & np.uint16(0xFF8F))
    assert masked.size == 1 << 13
    assert np.array_equal(np.sort(lut[masked]), np.arange(1 << 13))
    idx = A.MaskedIndex()
    a = idx(np.repeat(masked, 64), np.tile(masked[:: 128], masked.size))
    assert np.unique(a).size == a.size and int(a.max()) < idx.size
