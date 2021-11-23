import pytest

from src.settings import default_settings
from src.izgbs import izgbs, voting_time_calcs


def test_voting_time_calcs_default1():
    _min, _mode, _max = voting_time_calcs(10, default_settings())

    assert _min == 6 and _mode == 10 and _max == 20


def test_izgbs_full_run1():
    settings = default_settings()

    results = izgbs(
        200,
        100,
        1,
        {
            'Eligible Voters': 100,
            'Likely or Exp. Voters': 200,
            'Ballot Length Measure': 10
        },
        settings
    )

    assert type(results) == dict


def test_izgbs_atleast_one_feasible():
    settings = default_settings()

    results = izgbs(
        200,
        100,
        1,
        {
            'Eligible Voters': 100,
            'Likely or Exp. Voters': 200,
            'Ballot Length Measure': 10
        },
        settings
    )

    assert any(v['Feasible'] for v in results.values())


def test_izgbs_no_infeasible_after_first_feasible():
    settings = default_settings()

    results = izgbs(
        200,
        100,
        1,
        {
            'Eligible Voters': 100,
            'Likely or Exp. Voters': 200,
            'Ballot Length Measure': 10
        },
        settings
    )

    results = list(results.values())
    first_feasible = None

    for i, res in enumerate(results):
        if res['Feasible']:
            first_feasible = i
            break

    assert all(res['Feasible'] for res in results[first_feasible:])


def test_izgbs_can_return_all_infeasible():
    settings = default_settings()

    results = izgbs(
        3,
        2,
        1,
        {
            'Eligible Voters': 10000,
            'Likely or Exp. Voters': 20000,
            'Ballot Length Measure': 100
        },
        settings
    )

    assert not any(v['Feasible'] for v in results.values())


def test_izgbs_all_feasible_with_inf_service_req():
    settings = default_settings()

    settings['SERVICE_REQ'] = float('inf')

    results = izgbs(
        4,
        3,
        0,
        {
            'Eligible Voters': 10000,
            'Likely or Exp. Voters': 20000,
            'Ballot Length Measure': 100
        },
        settings
    )

    assert all(v['Feasible'] for v in results.values())


def test_izgbs_feasible_dict_size_determined_by_num_machines():
    settings = default_settings()

    upper = 400
    lower = 10
    results = izgbs(
        upper,
        11,
        lower,
        {
            'Eligible Voters': 10000,
            'Likely or Exp. Voters': 20000,
            'Ballot Length Measure': 100
        },
        settings
    )

    assert len(results) == upper - lower


def test_izgbs_feasible_dict_contains_correct_machine_nums():
    settings = default_settings()

    upper = 400
    lower = 10
    results = izgbs(
        upper,
        11,
        lower,
        {
            'Eligible Voters': 10000,
            'Likely or Exp. Voters': 20000,
            'Ballot Length Measure': 100
        },
        settings
    )

    assert list(results.keys()) == [*range(lower + 1, upper + 1)]


def test_izgbs_bad_start_raises_1():
    settings = default_settings()

    with pytest.raises(Exception):
        izgbs(
            400,
            3,
            10,
            {
                'Eligible Voters': 10000,
                'Likely or Exp. Voters': 20000,
                'Ballot Length Measure': 100
            },
            settings
        )


def test_izgbs_bad_start_raises_2():
    settings = default_settings()

    with pytest.raises(Exception):
        izgbs(
            400,
            403,
            10,
            {
                'Eligible Voters': 10000,
                'Likely or Exp. Voters': 20000,
                'Ballot Length Measure': 100
            },
            settings
        )
