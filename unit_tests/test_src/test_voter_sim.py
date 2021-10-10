import pytest
from statistics import mean

from src.voter_sim import voter_sim


def test_voter_sim_many_machines_1():
    # act
    wait_times = voter_sim(
        max_voters=100,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        arrival_rt=0.01,
        num_machines=100
    )
    # assert - checking that all voters get to vote if num_machines == num_voters
    assert len(wait_times) == 100


def test_voter_sim_many_machines_2():
    # act
    wait_times = voter_sim(
        max_voters=100,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        arrival_rt=0.01,
        num_machines=100
    )
    # assert - checking that the wait times are all 0.0 if num_machines == num_voters
    assert wait_times.count(0.0) == 100


def test_voter_sim_non_zero_wait_times():
    # act
    wait_times = voter_sim(
        max_voters=100,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        arrival_rt=0.01,
        num_machines=50
    )
    # assert - checking that the wait times are not all 0.0 if num_machines < num_voters
    assert wait_times.count(0.0) != 100


def test_voter_sim_zero_initial_wait_time():
    # act
    wait_times = voter_sim(
        max_voters=100,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        arrival_rt=0.01,
        num_machines=1
    )
    # assert - checking that the first wait time is zero
    assert wait_times[0] == 0.0


def test_voter_sim_zero_machines_raises():
    # act/assert
    with pytest.raises(Exception):
        _ = voter_sim(
            max_voters=100,
            vote_time_min=1,
            vote_time_mode=2,
            vote_time_max=3,
            arrival_rt=0.01,
            num_machines=0
        )


def test_voter_sim_usual_usage_1():
    # act
    wait_times = voter_sim(
        max_voters=914,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        arrival_rt=0.5,
        num_machines=7
    )
    # assert
    assert mean(wait_times) == 1.0
