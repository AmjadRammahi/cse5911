import logging
import numpy as np
import simpy as sp
from random import expovariate

import settings


class VotingLocation(object):
    def __init__(self, env, res_df, num_machines, vote_time_min, vote_time_mode, vote_time_max):
        self.env = env
        self.res_df = res_df
        self.vote_time_min = vote_time_min
        self.vote_time_mode = vote_time_mode
        self.vote_time_max = vote_time_max
        self.machine = sp.Resource(env, capacity=num_machines)

    def vote(self, voter):
        # randomly generate voting time
        voting_time = vote_time(self.vote_time_min, self.vote_time_mode, self.vote_time_max)

        yield self.env.timeout(voting_time)

        self.res_df.loc[self.res_df.Voter == voter, 'Voting_Time'] = voting_time

        logging.debug(f'{voter} voted in {voting_time} minutes.')


def vote_time(vote_time_min, vote_time_mode, vote_time_max):
    return np.random.triangular(vote_time_min, vote_time_mode, vote_time_max, size=None)


def generate_voter(arrival_rt):
    return expovariate(1.0 / arrival_rt)


def voter(
    env,
    res_df,
    name,
    voter_num,
    vm
):
    # Indicates that this row of the results df is used
    res_df.loc[res_df.Voter == name, 'Used'] = 1
    res_df.loc[res_df.Voter == name, 'Arrival_Time'] = env.now

    logging.debug(f'{name} arrives at the polls at {env.now:.2f}.')

    with vm.machine.request() as request:
        yield request

        res_df.loc[res_df.Voter == name, 'Voting_Start_Time'] = env.now

        logging.debug(f'{name} enters the polls at {env.now:.2f}.')

        yield env.process(vm.vote(name))

        res_df.loc[res_df.Voter == name, 'Departure_Time'] = env.now

        logging.debug(f'{name} leaves the polls at {env.now:.2f}.')


def setup(
    env,
    res_df,
    max_voter,
    vote_time_min,
    vote_time_mode,
    vote_time_max,
    arrival_rt,
    num_machines
):
    # create the voting machine
    voting_machine = VotingLocation(env, res_df, num_machines, vote_time_min, vote_time_mode, vote_time_max)

    # this logic prevents us from creating more than the specified maximum number of voters
    for i in range(max_voter):
        # generate arrival time for next voter
        t = generate_voter(arrival_rt)
        yield env.timeout(t)
        env.process(voter(env, res_df, f'Voter {i}', i, voting_machine))


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
        TODO.

        # TODO: move these params to os.environ.
        # Once done move these inner functions/class out to file scope - Collin
        Params:
            res_df () TODO,
            max_voter () TODO,
            vote_time_min () TODO,
            vote_time_mode () TODO,
            vote_time_max () TODO,
            arrival_rt () TODO,
            num_machines () TODO.

        Returns:
            TODO.
    '''
    RANDOM_SEED = 56  # for repeatability during testing
    SIM_TIME = (settings.POLL_END - settings.POLL_START) * 60  # simulation time in minutes

    # Setup and start the simulation

    # rand.seed(RANDOM_SEED)  # SAVE: This helps reproduce the results

    # create an environment and start the setup process
    env = sp.Environment()
    env.process(setup(env, res_df, num_machines, vote_time_min, vote_time_mode, vote_time_max, max_voter, arrival_rt))

    # environment will run until end of simulation time
    env.run(until=SIM_TIME)

    return res_df
