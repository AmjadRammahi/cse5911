import pytest

from src.izgbs import izgbs, voting_time_calcs


def test_voting_time_calcs_default1():
    _min, _mode, _max = voting_time_calcs(10)

    assert _min == 6 and _mode == 10 and _max == 20


def test_izgbs_full_run1():
    results = izgbs(
        200,
        100,
        1,
        1.0,
        {
            'Eligible Voters': 100,
            'Likely or Exp. Voters': 200,
            'Ballot Length Measure': 10
        },
        30.0
    )

    assert type(results) == dict


def test_izgbs_atleast_one_feasible():
    results = izgbs(
        200,
        100,
        1,
        1.0,
        {
            'Eligible Voters': 100,
            'Likely or Exp. Voters': 200,
            'Ballot Length Measure': 10
        },
        30.0
    )

    assert any(v['Feasible'] for v in results.values())


def test_izgbs_no_infeasible_after_first_feasible():
    results = izgbs(
        200,
        100,
        1,
        1.0,
        {
            'Eligible Voters': 100,
            'Likely or Exp. Voters': 200,
            'Ballot Length Measure': 10
        },
        30.0
    )

    results = list(results.values())
    first_feasible = None

    for i, res in enumerate(results):
        if res['Feasible']:
            first_feasible = i
            break

    assert all(res['Feasible'] for res in results[first_feasible:])


def test_izgbs_can_return_all_infeasible():
    results = izgbs(
        3,
        2,
        1,
        1.0,
        {
            'Eligible Voters': 10000,
            'Likely or Exp. Voters': 20000,
            'Ballot Length Measure': 100
        },
        30.0
    )

    assert not any(v['Feasible'] for v in results.values())


def test_izgbs_all_feasible_with_inf_service_req():
    results = izgbs(
        4,
        3,
        0,
        1.0,
        {
            'Eligible Voters': 10000,
            'Likely or Exp. Voters': 20000,
            'Ballot Length Measure': 100
        },
        float('inf')
    )

    assert all(v['Feasible'] for v in results.values())


def test_izgbs_feasible_dict_size_determined_by_num_machines():
    upper = 400
    lower = 10
    results = izgbs(
        upper,
        11,
        lower,
        1.0,
        {
            'Eligible Voters': 10000,
            'Likely or Exp. Voters': 20000,
            'Ballot Length Measure': 100
        },
        float('inf')
    )

    assert len(results) == upper - lower


def test_izgbs_feasible_dict_contains_correct_machine_nums():
    upper = 400
    lower = 10
    results = izgbs(
        upper,
        11,
        lower,
        1.0,
        {
            'Eligible Voters': 10000,
            'Likely or Exp. Voters': 20000,
            'Ballot Length Measure': 100
        },
        float('inf')
    )

    assert list(results.keys()) == [*range(lower + 1, upper + 1)]


def test_izgbs_bad_start_raises_1():
    with pytest.raises(Exception):
        izgbs(
            400,
            3,
            10,
            1.0,
            {
                'Eligible Voters': 10000,
                'Likely or Exp. Voters': 20000,
                'Ballot Length Measure': 100
            },
            float('inf')
        )


def test_izgbs_bad_start_raises_2():
    with pytest.raises(Exception):
        izgbs(
            400,
            403,
            10,
            1.0,
            {
                'Eligible Voters': 10000,
                'Likely or Exp. Voters': 20000,
                'Ballot Length Measure': 100
            },
            float('inf')
        )
