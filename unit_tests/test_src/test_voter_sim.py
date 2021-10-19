import pytest
from math import isclose
from statistics import mean
import simpy
from src.izgbs import voting_time_calcs
from src.voter_sim import voter_sim
from src.voter_sim import VotingLocation
from src.settings import Settings


# NOTE: can simulate N number of times to reduce variance
# NOTE: use math.isclose()

# General set-up for all voter_sim unit tests
# @pytest.fixture(scope="function", autouse=True)
# def test_test():
#     # TODO: determine if 'random.seed' actually improves run-to-run consistency
#     random_seed = 42
#     random.seed(random_seed)
#
#     max_voter = 2
#     vote_time_min = 2
#     vote_time_mode = 4
#     vote_time_max = 6
#     arrival_rt = 1
#     num_machines = 2
#
#     wait_times = voter_sim(
#         max_voters=2,
#         vote_time_min=2,
#         vote_time_mode=4,
#         vote_time_max=6,
#         arrival_rt=100 / 13 / 60,
#         num_machines=2
#     )
#
#     assert 1 == 1


# Because the "generate_voter" function only contains a random number generator, it
# cannot be tested.
def test_generate_voter_always_true():
    assert 1 == 1


# Check that the correct dictionary size is generated
def test_voting_location_init_dict_length():
    num_voters = 100
    location = VotingLocation(
        env=simpy.Environment(),
        max_voters=num_voters,
        num_machines=100,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        arrival_rt=100 / 13 / 60,
        sim_time=Settings.POLL_OPEN
    )

    assert len(location.voters_dict) == num_voters


def test_voter_sim_many_machines_1():
    # act
    wait_times = voter_sim(
        max_voters=100,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        arrival_rt=100 / 13 / 60,
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

# def test_voter_sim_usual_usage_1():
#     _min, _mode, _max = voting_time_calcs(5)
#     # act
#     wait_times = voter_sim(
#         max_voters=914,
#         vote_time_min=_min,
#         vote_time_mode=_mode,
#         vote_time_max=_max,
#         arrival_rt=914 / 13 / 60,
#         num_machines=7
#     )
#     # assert
#     assert mean(wait_times) == 1.0
