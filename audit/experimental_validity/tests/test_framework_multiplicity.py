from framework.multiplicity import holm_correction


def test_holm_correction_known_values():
    """
    CASE 7: multiple comparisons - verify against a hand-computed Holm
    step-down result for p = [0.01, 0.04, 0.03, 0.005], alpha=0.05.

    Sorted ascending: 0.005 (i=4), 0.01 (i=3), 0.03 (i=2), 0.04 (i=1)
    (m=4 hypotheses; k=rank)
      k=1: (4-1+1)*0.005 = 4*0.005 = 0.020
      k=2: (4-2+1)*0.01  = 3*0.01  = 0.030
      k=3: (4-3+1)*0.03  = 2*0.03  = 0.060 -> running max stays 0.060
      k=4: (4-4+1)*0.04  = 1*0.04  = 0.040 -> running max = max(0.060,0.040)=0.060
    """
    result = holm_correction(
        family_id="test_family",
        hypothesis_ids=["h1", "h2", "h3", "h4"],
        p_values=[0.01, 0.04, 0.03, 0.005],
        alpha=0.05,
    )
    adj = dict(zip(result.hypothesis_ids, result.adjusted_p_values))
    assert abs(adj["h4"] - 0.020) < 1e-9
    assert abs(adj["h1"] - 0.030) < 1e-9
    assert abs(adj["h3"] - 0.060) < 1e-9
    assert abs(adj["h2"] - 0.060) < 1e-9

    reject = dict(zip(result.hypothesis_ids, result.reject_null))
    assert reject["h4"] is True   # 0.020 < 0.05
    assert reject["h1"] is True   # 0.030 < 0.05
    assert reject["h3"] is False  # 0.060 >= 0.05
    assert reject["h2"] is False  # 0.060 >= 0.05


def test_holm_correction_handles_none_p_values():
    result = holm_correction(
        family_id="test_family_with_missing",
        hypothesis_ids=["h1", "h2"],
        p_values=[0.01, None],
        alpha=0.05,
    )
    idx_h2 = result.hypothesis_ids.index("h2")
    assert result.adjusted_p_values[idx_h2] is None
    assert result.reject_null[idx_h2] is None


def test_holm_correction_two_test_primary_family():
    """The actual frozen primary family: {H-EV-SHUFFLE, H-EV-REPRESENTATION}."""
    result = holm_correction(
        family_id="ev_primary_family_v1",
        hypothesis_ids=["H-EV-SHUFFLE", "H-EV-REPRESENTATION"],
        p_values=[0.03, 0.20],
        alpha=0.05,
    )
    shuffle = result.for_hypothesis("H-EV-SHUFFLE")
    repr_ = result.for_hypothesis("H-EV-REPRESENTATION")
    # smaller p (0.03) gets multiplied by 2 -> 0.06 (not significant after correction)
    assert abs(shuffle["adjusted_p_value"] - 0.06) < 1e-9
    assert shuffle["reject_null"] is False
    assert abs(repr_["adjusted_p_value"] - 0.20) < 1e-9
    assert repr_["reject_null"] is False
