import pytest
import simpy
from src.settings import default_settings
from src.voter_sim import VotingLocation, voter_sim


# TODO: can simulate N number of times to reduce variance


def test_voting_location_init_dict_length():
    num_voters = 100
    location = VotingLocation(
        env=simpy.Environment(),
        max_voters=num_voters,
        expected_voters=num_voters,
        num_machines=100,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        sim_time=13
    )
    # assert - checking that the correct dictionary size is generated
    # (size == num_voters)
    assert len(location.voters_dict) == num_voters


def test_voter_sim_many_machines():
    settings = default_settings()

    # act
    num_voters = 100
    wait_times = voter_sim(
        max_voters=num_voters,
        expected_voters=num_voters,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        num_machines=100,
        settings=settings
    )
    # assert - checking that the wait times are all 0.0 if num_machines == num_voters
    assert wait_times.count(0.0) == len(wait_times)


def test_voter_sim_non_zero_wait_times():
    settings = default_settings()

    # act
    # voters will be equal to the simulation time (in minutes) which will
    # guarantee that voters will be forced to wait in line because there is
    # only one machine.
    sim_time = 6.5 * 60
    wait_times = voter_sim(
        max_voters=int(sim_time),
        expected_voters=int(sim_time),
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        num_machines=1,
        settings=settings
    )
    # assert - checking that the wait times are not all 0.0 if num_machines << num_voters
    assert wait_times.count(0.0) != 100


def test_voter_sim_zero_initial_wait_time():
    settings = default_settings()

    # act
    wait_times = voter_sim(
        max_voters=100,
        expected_voters=100,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        num_machines=1,
        settings=settings
    )
    # assert - checking that the first wait time is zero
    assert wait_times[0] == 0.0


def test_voter_sim_zero_machines_raises():
    # act/assert
    with pytest.raises(Exception):
        _ = voter_sim(
            max_voters=100,
            expected_voters=100,
            vote_time_min=1,
            vote_time_mode=2,
            vote_time_max=3,
            num_machines=0
        )


def test_voter_sim_near_expected_should_vote():
    settings = default_settings()

    # Override settings and have the polls open for only one hour
    # max_voters = number of minutes the polls are open multiplied by 16 which
    # is approximately double the arrival rate. This guarantees that under the
    # not all voters will vote.
    max_voters = int(1.0 * 60 * 16)
    # expected_voters will be set to half of the maximum which is similar to
    # most locations in the original Excel file.
    expected_voters = int(max_voters / 2)
    wait_times = voter_sim(
        max_voters=max_voters,
        expected_voters=expected_voters,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        num_machines=10,
        settings=settings
    )
    # assert - the number of actual voters should be near (+-!0%) the number of
    # expected voters
    assert len(wait_times) == pytest.approx(expected_voters, rel=0.1)


def test_generate_voter_random_arrival():
    num_voters = 200
    sim_time = 60
    env = simpy.Environment()
    location = VotingLocation(
        env=env,
        max_voters=num_voters,
        expected_voters=num_voters,
        num_machines=10,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        sim_time=sim_time
    )
    # First arrival is special because it does not depend on the previous
    # voter(s)
    arrival_times = [VotingLocation.generate_voter(location)]
    # Get remaining arrival times
    for i in range(1, num_voters):
        # Because SimPy adds voters one at a time the arrival time of any
        # given voter is equal to the randomly generated time plus the arrival
        # time of the previous voter.
        arrival_times.append(arrival_times[i - 1] + VotingLocation.generate_voter(location))
    # Arrival times should be randomly distributed, therefore the average
    # arrival time should be in the middle of poll's hours of operation.
    optimal_avg_arrival = sim_time / 2
    actual_avg_arrival = 0.0
    # Take the sum of all arrival times
    for arrival in arrival_times:
        actual_avg_arrival = actual_avg_arrival + arrival
    # Calculate average
    actual_avg_arrival = actual_avg_arrival / num_voters
    # assert - the actual average arrival time should be within 10% of the
    # optimal average
    assert actual_avg_arrival == pytest.approx(optimal_avg_arrival, rel=0.20), \
        "Observed average arrival times do not match expected random arrival times."


def test_generate_voter_random_arrival_2():
    # Many more total voters than the previous test. Designed to break the
    # arrival times with a very late average time.
    num_voters = int(6.5 * 60 * 16)
    sim_time = 60
    env = simpy.Environment()
    location = VotingLocation(
        env=env,
        max_voters=num_voters,
        expected_voters=num_voters,
        num_machines=10,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        sim_time=sim_time
    )
    # First arrival is special because it does not depend on the previous
    # voter(s)
    arrival_times = [VotingLocation.generate_voter(location)]
    # Get remaining arrival times
    for i in range(1, num_voters):
        # Because SimPy adds voters one at a time the arrival time of any
        # given voter is equal to the randomly generated time plus the arrival
        # time of the previous voter.
        arrival_times.append(arrival_times[i - 1] + VotingLocation.generate_voter(location))
    # Arrival times should be randomly distributed, therefore the average
    # arrival time should be in the middle of poll's hours of operation.
    optimal_avg_arrival = sim_time / 2
    actual_avg_arrival = 0.0
    # Take the sum of all arrival times
    for arrival in arrival_times:
        actual_avg_arrival = actual_avg_arrival + arrival
    # Calculate average
    actual_avg_arrival = actual_avg_arrival / num_voters
    # assert - the actual average arrival time should be within 10% of the
    # optimal average
    assert actual_avg_arrival == pytest.approx(optimal_avg_arrival, rel=0.1), \
        "Observed average arrival times do not match expected random arrival times."


def test_voter_sim_total_voters_must_not_exceed_max():
    settings = default_settings()

    max_voters = 1
    wait_times = voter_sim(
        max_voters=max_voters,
        expected_voters=max_voters * 10,
        vote_time_min=1,
        vote_time_mode=2,
        vote_time_max=3,
        num_machines=1,
        settings=settings
    )
    # assert - total number of voters should never exceed the maximum number of
    # registered voters
    assert len(wait_times) <= max_voters
