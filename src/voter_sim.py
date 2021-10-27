import simpy
import logging
import numpy as np
from random import expovariate
from numba import njit
from src.settings import Settings


# NOTE: this is TandemQueueWQuartile


class VotingLocation(object):
    def __init__(
            self,
            env: simpy.Environment,
            max_voters: int,
            expected_voters: int,
            num_machines: int,
            vote_time_min: float,
            vote_time_mode: float,
            vote_time_max: float,
            sim_time: float
    ):
        self.env = env
        self.voters_dict = {
            f'Voter {i}': {}
            for i in range(max_voters)
        }
        self.max_voters = max_voters
        self.expected_voters = expected_voters
        self.vote_time_min = vote_time_min
        self.vote_time_mode = vote_time_mode
        self.vote_time_max = vote_time_max
        self.voting_machines = simpy.Resource(env, capacity=num_machines)
        self.sim_time = sim_time
        self.arrival_rt = self.calc_arrival_rate()

        self.wait_times = []

        # kick off the simulation
        self.env.process(self.run())

    def run(self) -> None:
        '''
            Runs the simulation at this VotingLocation.
            Processes voters until the simulation ends.

            Returns:
                None.
        '''
        for i in range(self.max_voters):
            # New voters cannot get in line after the polls close
            if self.env.now >= self.sim_time:
                break
            # generate arrival time for next voter
            t = self.generate_voter()
            yield self.env.timeout(t)

            self.env.process(
                self.voter(f'Voter {i}')
            )

    def generate_voter(self) -> float:
        '''
            Returns a float to represent the arrival time of the next voter
            from an exponential distribution.
            https://en.wikipedia.org/wiki/Exponential_distribution

            Returns:
                (float) : voter iter-arrival time.
        '''
        return expovariate(1.0 / self.arrival_rt)

    def calc_arrival_rate(self):
        '''
            Calculates an average arrival rate which will allow the expected
            number of voters to arrive at the polling location before it
            closes. This average rate is based on the number of hours the
            polling location is open and the expected number of voters.

        Returns:
            (float) : average voter arrival rate
        '''
        return self.sim_time / self.expected_voters

    def voter(self, name: str) -> None:
        '''
            Marks a potential voter as having voted.
            Invokes the vote() function so that the voter 'votes'.

            Params:
                name (str) : name of the voter, ex: 'Voter 0'.

            Returns:
                None.
        '''
        self.voters_dict[name]['Arrival_Time'] = self.env.now

        logging.debug(f'{name} arrives at the polls at {self.env.now:.2f}.')

        # NOTE: this is like a mutex lock, the voter is requesting a voting machine
        # https://simpy.readthedocs.io/en/latest/api_reference/simpy.resources.html#simpy.resources.resource.Resource.request
        with self.voting_machines.request() as request:
            yield request

            self.voters_dict[name]['Voting_Start_Time'] = self.env.now

            self.wait_times.append(
                self.voters_dict[name]['Voting_Start_Time'] - self.voters_dict[name]['Arrival_Time']
            )

            logging.debug(f'{name} enters the polls at {self.env.now:.2f}.')

            yield self.env.process(self.vote(name))

            self.voters_dict[name]['Departure_Time'] = self.env.now

            logging.debug(f'{name} leaves the polls at {self.env.now:.2f}.')

    def vote(self, name: str) -> None:
        '''
            Updates the voter with it's voting time.
            Yields the voting time as a timeout.

            Params:
                name (str) : name of the voter, ex: 'Voter 0'.

            Returns:
                None.
        '''
        # randomly generate voting time
        voting_time = self.vote_time()

        yield self.env.timeout(voting_time)

        self.voters_dict[name]['Voting_Time'] = voting_time

        logging.debug(f'{name} voted in {voting_time} minutes.')

    def vote_time(self) -> float:
        '''
            Returns a random voting time within a triangular distribution.
            https://en.wikipedia.org/wiki/Triangular_distribution

            Returns:
                (float) : voting time.
        '''
        return np.random.triangular(
            self.vote_time_min,
            self.vote_time_mode,
            self.vote_time_max,
            size=None
        )


def voter_sim(
        *,
        max_voters: int,
        expected_voters: int,
        vote_time_min: float,
        vote_time_mode: float,
        vote_time_max: float,
        arrival_rt: float,
        num_machines: int
) -> list:
    '''
        Executes a voting simulation given various inputs.

        Params:
            max_voters (int) : maximum number of voters,
            expected_voters (int) : the likely (or expected) number of voters,
            vote_time_min (float) : min voting time,
            vote_time_mode (float) : mode voting time,
            vote_time_max (float) : max voting time,
            arrival_rt (float) : arrival mean,
            num_machines (int) : number of voting machines at the location.

        Returns:
            (list) : wait times.
    '''

    sim_time = Settings.POLL_OPEN * 60

    # create an environment and start the setup process
    env = simpy.Environment()

    # create the voting location
    location = VotingLocation(
        env,
        max_voters,
        expected_voters,
        num_machines,
        vote_time_min,
        vote_time_mode,
        vote_time_max,
        sim_time
    )

    # environment will run until all voters have finished voting
    env.run()

    return location.wait_times
