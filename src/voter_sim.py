import simpy
import logging
import numpy as np
from random import expovariate

from src.settings import Settings


class VotingLocation(object):
    def __init__(
        self,
        env: simpy.Environment,
        results_dict: dict,
        max_voter: int,
        num_machines: int,
        vote_time_min: float,
        vote_time_mode: float,
        vote_time_max: float,
        arrival_rt: float
    ):
        self.env = env
        self.results_dict = results_dict
        self.max_voter = max_voter
        self.vote_time_min = vote_time_min
        self.vote_time_mode = vote_time_mode
        self.vote_time_max = vote_time_max
        self.arrival_rt = arrival_rt
        self.voting_machines = simpy.Resource(env, capacity=num_machines)

        # kick off the simulation
        self.env.process(self.run())

    def run(self) -> None:
        '''
            Runs the simulation at this VotingLocation.
            Processes voters until the simulation ends.

            Returns:
                None.
        '''
        for i in range(self.max_voter):
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
                (float) : voter arrival time.
        '''
        return expovariate(1.0 / self.arrival_rt)

    def voter(self, name: str) -> None:
        '''
            Marks a potential voter as having voted.
            Invokes the vote() function so that the voter 'votes'.

            Params:
                name (str) : name of the voter, ex: 'Voter 0'.

            Returns:
                None.
        '''
        self.results_dict[name]['Arrival_Time'] = self.env.now

        logging.debug(f'{name} arrives at the polls at {self.env.now:.2f}.')

        # NOTE: this is like a mutex lock, the voter is requesting a voting machine
        # https://simpy.readthedocs.io/en/latest/api_reference/simpy.resources.html#simpy.resources.resource.Resource.request
        with self.voting_machines.request() as request:
            yield request

            self.results_dict[name]['Voting_Start_Time'] = self.env.now

            logging.debug(f'{name} enters the polls at {self.env.now:.2f}.')

            yield self.env.process(self.vote(name))

            self.results_dict[name]['Departure_Time'] = self.env.now

            logging.debug(f'{name} leaves the polls at {self.env.now:.2f}.')

        self.results_dict[name]['Used'] = True

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

        self.results_dict[name]['Voting_Time'] = voting_time

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
    results_dict: dict,
    max_voter: int,
    vote_time_min: float,
    vote_time_mode: float,
    vote_time_max: float,
    arrival_rt: float,
    num_machines: int
) -> None:
    '''
        Executes a voting simulation given various inputs.

        Params:
            results_dict (dict) : DataFrame for sim results,
            max_voter (int) : maximum number of voters,
            vote_time_min (float) : min voting time,
            vote_time_mode (float) : mode voting time,
            vote_time_max (float) : max voting time,
            arrival_rt (float) : arrival mean,
            num_machines (int) : nuber of voting machines at the location.

        Returns:
            None.
    '''
    # RANDOM_SEED = 56  # for repeatability during testing
    # rand.seed(RANDOM_SEED)

    SIM_TIME = (Settings.POLL_END - Settings.POLL_START) * 60

    # create an environment and start the setup process
    env = simpy.Environment()

    # create the voting location
    VotingLocation(
        env,
        results_dict,
        max_voter,
        num_machines,
        vote_time_min,
        vote_time_mode,
        vote_time_max,
        arrival_rt
    )

    # environment will run until end of simulation time
    env.run(until=SIM_TIME)