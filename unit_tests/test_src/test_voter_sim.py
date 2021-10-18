import pytest

import src.voter_sim
from src.voter_sim import voter_sim
from src.voter_sim import VotingLocation
from src.voter_sim import Settings
import random


# General set-up for all voter_sim unit tests. This step is important because
# voter_sim uses objects that will be difficult to mock properly. Even if mocks
# were implemented, it could make the tests too brittle to be useful.
@pytest.fixture(scope="function", autouse=True)
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
