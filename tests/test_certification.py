from prep.certification import certificate_total_sample, direction_deltas, residual_upper


def test_joint_budget_split_and_single_sign():
    assert direction_deltas(10, 10, 0.05) == (0.025, 0.025)
    assert direction_deltas(10, 0, 0.05)[0] == 0.05
    assert direction_deltas(0, 10, 0.05)[1] == 0.05


def test_census_has_no_residual_uncertainty():
    assert residual_upper(12, 12, 3, 0.025) == 0
    result = certificate_total_sample(
        t_plus=6, t_minus=6, n=12, n_plus=6,
        e_plus=0, e_minus=0, c0=0, alpha_query=0.05,
    )
    assert result.status == "CENSUS_ZERO_RETAINED_BENEFIT"

