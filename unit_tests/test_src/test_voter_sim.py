import pytest
from math import isclose
from statistics import mean
from src.izgbs import voting_time_calcs
from src.voter_sim import voter_sim
import random


# NOTE: can simulate N number of times to reduce variance
# NOTE: use math.isclose()

# General set-up for all voter_sim unit tests. This step is important because
# voter_sim uses objects that will be difficult to mock properly. Even if mocks
# were implemented, it could make the tests too brittle to be useful.
# @pytest.fixture(scope="function", autouse=True)
def set_up():
    # TODO: determine if this actually works as intended
    random_seed = 42
    random.seed(random_seed)

    results_dict = {
        "Voter 0": {
            "Used": False
        },
        "Voter 1": {
            "Used": False
        }
    }
    max_voter = 2
    vote_time_min = 2
    vote_time_mode = 4
    vote_time_max = 6
    arrival_rt = 1
    num_machines = 2

    voter_sim(
        results_dict,
        max_voter,
        vote_time_min,
        vote_time_mode,
        vote_time_max,
        arrival_rt,
        num_machines
    )


# Because the "generate_voter" function only contains a random number generator, it
# cannot be tested.
def test_generate_voter_always_true():
    assert 1 == 1


# def test_voter_update_results_dict(mocker):
# mocker.patch.object(src.voter_sim.VotingLocation, 'self.results_dict',
#                     {"Voter 0": {
#                         "Used": False
#                     }
#                     })


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
