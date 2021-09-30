import simpy
import logging
import numpy as np
from random import expovariate

import settings


class VotingLocation(object):
    def __init__(
        self,
        env,
        res_df,
        max_voter,
        num_machines,
        vote_time_min,
        vote_time_mode,
        vote_time_max,
        arrival_rt
    ):
        self.env = env
        self.res_df = res_df
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
        # indicates that this row of the results df is used
        self.res_df.loc[self.res_df.Voter == name, 'Used'] = 1
        self.res_df.loc[self.res_df.Voter == name, 'Arrival_Time'] = self.env.now

        logging.debug(f'{name} arrives at the polls at {self.env.now:.2f}.')

        # NOTE: this is like a mutex lock, the voter is requesting a voting machine
        # https://simpy.readthedocs.io/en/latest/api_reference/simpy.resources.html#simpy.resources.resource.Resource.request
        with self.voting_machines.request() as request:
            yield request

            self.res_df.loc[self.res_df.Voter == name, 'Voting_Start_Time'] = self.env.now

            logging.debug(f'{name} enters the polls at {self.env.now:.2f}.')

            yield self.env.process(self.vote(name))

            self.res_df.loc[self.res_df.Voter == name, 'Departure_Time'] = self.env.now

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

        self.res_df.loc[self.res_df.Voter == name, 'Voting_Time'] = voting_time

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
    res_df,
    max_voter,
    vote_time_min,
    vote_time_mode,
    vote_time_max,
    arrival_rt,
    num_machines
):
    '''
        Executes a voting simulation given various inputs.

        Params:
            res_df (pd.DataFrame) : DataFrame for sim results,
            max_voter (int) : maximum number of voters,
            vote_time_min (float) : min voting time,
            vote_time_mode (float) : mode voting time,
            vote_time_max (float) : max voting time,
            arrival_rt (float) : arrival mean,
            num_machines (int) : nuber of voting machines at the location.

        Returns:
            (pd.DataFrame) : simulation results.
    '''
    # RANDOM_SEED = 56  # for repeatability during testing
    # rand.seed(RANDOM_SEED)

    SIM_TIME = (settings.POLL_END - settings.POLL_START) * 60  # simulation time in minutes

    # create an environment and start the setup process
    env = simpy.Environment()

    # create the voting location
    voting_location = VotingLocation(
        env,
        res_df,
        max_voter,
        num_machines,
        vote_time_min,
        vote_time_mode,
        vote_time_max,
        arrival_rt
    )

    # environment will run until end of simulation time
    env.run(until=SIM_TIME)

    return voting_location.res_df
