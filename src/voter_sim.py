import os  # TODO: use this to clean up voter_sim - Collin
import logging
import numpy as np
import simpy as sp
from random import expovariate


def voter_sim(
    res_df,
    max_voter,
    vote_time_min,
    vote_time_mode,
    vote_time_max,
    per_start,
    per_end,
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
            per_start () TODO,
            per_end () TODO,
            arrival_rt () TODO,
            num_machines () TODO.

        Returns:
            TODO.
    '''
    RANDOM_SEED = 56  # for repeatability during testing
    SIM_TIME = (per_end - per_start) * 60  # simulation time in minutes

    def vote_time():
        return np.random.triangular(vote_time_min, vote_time_mode, vote_time_max, size=None)

    def generate_voter():
        return expovariate(1.0 / arrival_rt)

    class VotingLocation(object):
        def __init__(self, env, num_machines):
            self.env = env
            self.machine = sp.Resource(env, capacity=num_machines)

        def vote(self, voter):
            # randomly generate voting time
            voting_time = vote_time()

            yield self.env.timeout(voting_time)

            res_df.loc[res_df.Voter == voter, 'Voting_Time'] = voting_time

            logging.debug(f'{voter} voted in {voting_time} minutes.')

    def voter(
        env,
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
        num_machines,
        arrival_rt,
        max_voters
    ):
        # create the voting machine
        voting_machine = VotingLocation(env, num_machines)

        # this logic prevents us from creating more than the specified maximum number of voters
        for i in range(max_voter):
            # generate arrival time for next voter
            t = generate_voter()
            yield env.timeout(t)
            env.process(voter(env, f'Voter {i}', i, voting_machine))

    # Setup and start the simulation

    # rand.seed(RANDOM_SEED)  # SAVE: This helps reproduce the results

    # create an environment and start the setup process
    env = sp.Environment()
    env.process(setup(env, num_machines, arrival_rt, max_voter))

    # environment will run until end of simulation time
    env.run(until=SIM_TIME)

    return res_df
