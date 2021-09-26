import xlrd
import math
import logging
import argparse
import numpy as np
import pandas as pd
import scipy.stats as st

from src.util import set_logging_level
from src.voter_sim import voter_sim

parser = argparse.ArgumentParser()
parser.add_argument('input_xlsx', type=str,
                    help='first positional argument, input xlsx filepath')
parser.add_argument('--log', default='info', help='log level, ex: --log debug')

logger = logging.getLogger()


# ===============================================================================
# Globals

# Placeholder for input parameters
# Options: Apportionment, Allocation
prob_type = 'apportionment'  # NOTE: unused
# If apportionment, this is the maximum total that can be purchased
# If allocation, this is the total number of machines available to be allocated
MAX_MACHINES = 60

# General variables
# first and last voting locations. This allows results to be calculated for a subset of location
FIRST_LOC = 1
LAST_LOC = 5
NUM_LOCATIONS = LAST_LOC - FIRST_LOC + 1

time_delta = 0.5  # NOTE: unused - time increment in minutes
ALPHA_VALUE = 0.05  # Probability of rejecting the null hypothesis
DELTA_IZ = 0.5  # Indifference zone parameter
NUM_REPLICATIONS = 100

# deluxe version variable placeholders
# 5:30, 6:30, 7:30
EARLY_START = 5.5  # NOTE: effectively unused
POLL_START = 6.5
POLL_END = 19.5
Poll_Hours = (POLL_END - POLL_START) * 24  # NOTE: unused
EarlyVoterHours = POLL_START - EARLY_START

# waiting time of voter who waits the longest
SERVICE_REQ = 30

# Batch size
BATCH_SIZE = 20

# Specifies whether there is a minimum allocation requirement, and if so what it is
# Check that this is being used correctly
MIN_ALLOC_FLG = 1
MIN_ALLOC = 4
# min_alloc = 175

# Defining objective
# Current state: only 'Max', future state to include 'Quantile' and 'Average'
Objective = 'Max'  # NOTE: unused
# Not used in MVP
# ObjectiveQuantileValue = 0.95
# Waiting time <= how many minutes
mu0Value = 500  # NOTE unused

# Arrival time periods
# not currently used
# TotalNoOfPeriods = 4

# Voting times
MIN_VOTING_MIN = 6
MIN_VOTING_MODE = 8
MIN_VOTING_MAX = 12
MIN_BALLOT = 0
MAX_VOTING_MIN = 6
MAX_VOTING_MODE = 10
MAX_VOTING_MAX = 20
MAX_BALLOT = 10
# Only single resource has been built
NumberOfResources = 1  # NOTE: unused

# Not yet used
# check_in_times = [0.9, 1.1, 2.9]

# Calculate number of batches
NUM_BATCHES = NUM_REPLICATIONS / BATCH_SIZE

# Add voter average arrivals in minutes
POLL_OPEN = POLL_END - POLL_START

# Create results arrays
avgResources = np.zeros(NUM_LOCATIONS)  # NOTE: unused
avgWaitingTime = np.zeros(NUM_LOCATIONS)  # NOTE: unused
MaxWaitingTime = np.zeros(NUM_LOCATIONS)  # NOTE: unused
QuantWaitingTime = np.zeros(NUM_LOCATIONS)  # NOTE: unused
WaitProbabilities = np.zeros(NUM_LOCATIONS)  # NOTE: unused
MeanClosingTimes = np.zeros(NUM_LOCATIONS)  # NOTE: unused

# machine_df contains estimated voters, max voters, and ballot length

# TODO: this num_tests is heavily used and it looks like Tian had a TODO here for it - Collin
# placeholder for algorithm to determine next configuration to test. This should be moved to location loop.
num_test = 2

# Iterate over locations
loc_sol = np.zeros(NUM_LOCATIONS)  # NOTE: unused - number machines
loc_waits = np.zeros(NUM_LOCATIONS)  # NOTE: unused - voter wait times
loc_ct = np.zeros(NUM_LOCATIONS)  # NOTE: unused

# ===============================================================================
# Utility Functions


def voting_time_calcs(ballot):
    vote_min = \
        MIN_VOTING_MIN + \
        (MAX_VOTING_MIN - MIN_VOTING_MIN) / \
        (MAX_BALLOT - MIN_BALLOT) * \
        (ballot - MIN_BALLOT)

    vote_mode = \
        MIN_VOTING_MODE + \
        (MAX_VOTING_MODE - MIN_VOTING_MODE) / \
        (MAX_BALLOT - MIN_BALLOT) * \
        (ballot - MIN_BALLOT)

    vote_max = \
        MIN_VOTING_MAX + \
        (MAX_VOTING_MAX - MIN_VOTING_MAX) / \
        (MAX_BALLOT - MIN_BALLOT) * \
        (ballot - MIN_BALLOT)

    vote_avg = (vote_min + vote_mode + vote_max) / 3

    return vote_min, vote_mode, vote_max, vote_avg


def create_res_df(max_voters):
    '''
        This function is used to create an empty dataframe to store simulation results
    '''
    res_cols = [
        'Replication',
        'Voter',
        'Arrival_Time',
        'Voting_Start_Time',
        'Voting_Time',
        'Departure_Time',
        'Waiting_Time',
        'Total_Time',
        'Used'
    ]
    # Results dataframe with a record for the max possible number of voters
    voter_cols = np.zeros((max_voters, len(res_cols)))
    voter_results = pd.DataFrame(voter_cols, columns=res_cols)
    # Populates the voter ID field
    voter_results['Voter'] = 'Voter ' + voter_results.index.astype('str')

    return voter_results


def create_loc_df(vote_locs):
    '''
        This function creates a dataframe of IZGBS results to be outputted
    '''
    res_cols = ['Resource', 'Exp. Avg. Wait Time', 'Exp. Max. Wait Time']
    # Create an empty dataframe the same size as the locations dataframe
    voter_cols = np.zeros((vote_locs, len(res_cols)))
    loc_results = pd.DataFrame(voter_cols, columns=res_cols)
    # Populates the location ID field
    loc_results['Locations'] = (loc_results.index + 1).astype('str')

    return loc_results


def create_hypothesis_df(num_h):
    '''
        This function creates a dataframe to store hypothesis testing results
    '''
    # This dataframe will contain 1 record per machine count
    res_cols = ['Machines', 'Feasible', 'BatchAvg', 'BatchAvgMax']
    # Create an empty dataframe the same size as the locations dataframe
    voter_cols = np.zeros((num_h, len(res_cols)))
    hyp_results = pd.DataFrame(voter_cols, columns=res_cols)
    # Populates the Machine count field
    hyp_results['Machines'] = (hyp_results.index + 1).astype('int')

    return hyp_results


# ===============================================================================
# Main IZGBS Function


# NOTE: this appears to be the main izgbs function
def izgbs(
    vote_loc,
    total_num,
    start,
    min_val,
    alpha
):
    '''
        Main IZGBS function. TODO: fill out function desc

        Params:
            vote_loc () : asd.
            total_num () : asd.
            start () : asd.
            min_val () : asd.
            alpha () : asd.

        Returns:
            () : idk.
    '''
    # Read in parameters from locations dataframe
    # Voters = voting_machines['Likely or Exp. Voters'].loc[(voting_machines.Loc_ID==vote_loc)][0]
    # voters_max= voting_machines['Eligible Voters'].loc[(voting_machines.Loc_ID==vote_loc)][0]
    # ballot_lth = voting_machines['Ballot Length Measure'].loc[(voting_machines.Loc_ID==vote_loc)][0]
    # arrival_rt = voting_machines['Arrival_mean'].loc[(voting_machines.Loc_ID==vote_loc)][0]
    Voters = voting_machines['Likely or Exp. Voters'].loc[vote_loc]
    voters_max = voting_machines['Eligible Voters'].loc[vote_loc]
    ballot_lth = voting_machines['Ballot Length Measure'].loc[vote_loc]
    arrival_rt = voting_machines['Arrival_mean'].loc[vote_loc]

    # Calculate voting times
    vote_min, vote_mode, vote_max, vote_avg = voting_time_calcs(ballot_lth)

    # Create a dataframe for total number of machines
    feasible_df = create_hypothesis_df(total_num)

    # Start with the start value specified
    hyps_rem = 1
    num_test = start
    cur_lower = min_val
    cur_upper = total_num

    while (hyps_rem):
        print(f'Num Test: {num_test}')
        print(f'Current Upper: {cur_upper}')
        print(f'Current Lower {cur_lower}')

        first_rep = 1

        for i in range(1, NUM_REPLICATIONS):
            results_df = create_res_df(voters_max)

            # calculate voting times
            results_df = voter_sim(
                results_df,
                voters_max,
                vote_min,
                vote_mode,
                vote_max,
                POLL_START,
                POLL_END,
                arrival_rt,
                num_test,
                loud=0
            )

            # only keep results records that were actually used
            results_df = results_df[results_df['Used'] == 1].copy()
            results_df['Replication'] = i

            if (first_rep == 1):
                final_loc_results = results_df.copy()
                first_rep = 0
            else:
                final_loc_results = pd.concat([final_loc_results, results_df])

        # calculate waiting time
        # need to drop voters that did not finish
        final_loc_results['Waiting_Time'] = \
            final_loc_results['Voting_Start_Time'] - final_loc_results['Arrival_Time']

        final_loc_results['Total_Time'] = \
            final_loc_results['Departure_Time'] - final_loc_results['Arrival_Time']

        # calculate average and max waiting time by replication
        rep_nums = final_loc_results.groupby(
            ['Replication']
        )['Waiting_Time'].agg(['max', 'mean']).reset_index()

        # rename fields
        rep_nums.rename(
            columns={'max': 'Max_Waiting_Time', 'mean': 'Avg_Waiting_Time'},
            inplace=True
        )

        # assign records to batches
        rep_nums['Batch'] = rep_nums['Replication'] % NUM_BATCHES
        rep_nums.head()

        # calculate batch averages
        calcs = ['Max_Waiting_Time', 'Avg_Waiting_Time']
        batch_avgs = rep_nums.groupby(
            'Batch'
        )[['Max_Waiting_Time', 'Avg_Waiting_Time']].agg('mean').reset_index()

        avg_avg = batch_avgs['Avg_Waiting_Time'].mean()
        max_avg = batch_avgs['Max_Waiting_Time'].mean()
        max_std = batch_avgs['Max_Waiting_Time'].std()

        batch_avgs.head()
        print(avg_avg, max_avg, max_std)

        # populate results
        feasible_df.loc[feasible_df.Machines == num_test, 'BatchAvg'] = avg_avg
        feasible_df.loc[feasible_df.Machines == num_test, 'BatchAvgMax'] = max_avg

        # calculate test statistic
        if (max_std > 0):
            z = (max_avg - (SERVICE_REQ + DELTA_IZ)) / \
                (max_std / math.sqrt(NUM_BATCHES))
            p = st.norm.cdf(z)

            # feasible
            if (p < SASalphaValue):
                feasible_df.loc[feasible_df.Machines >= num_test, 'Feasible'] = 1
                print('scenario 1')
                # Move to lower half
                cur_upper = num_test
                t = math.floor((cur_upper - cur_lower) / 2)
                num_test = math.floor((cur_upper - cur_lower) / 2) + cur_lower
            else:
                print('scenario2')
                # Move to upper half
                feasible_df.loc[feasible_df.Machines == num_test, 'Feasible'] = 0
                cur_lower = num_test
                num_test = math.floor((cur_upper - num_test) / 2) + cur_lower
            # Declare feasible if Std = 0??
        else:
            print('scenario3')
            feasible_df.loc[feasible_df.Machines >= num_test, 'Feasible'] = 1
            # Move to lower half
            cur_upper = num_test
            t = math.floor((cur_upper - cur_lower) / 2)
            num_test = math.floor((cur_upper - cur_lower) / 2) + cur_lower
        # Check if there are hypotheses left to test
        # if (num_test >= min_val):
        #   hyps_rem=1
        # else:
        #   hyps_rem=0

        if (cur_upper > cur_lower and num_test > cur_lower and num_test < cur_upper):
            hyps_rem = 1
        else:
            hyps_rem = 0

    print(feasible_df)

    return feasible_df


def fetch_location_data(voting_config: xlrd.Book) -> list:
    location_sheet = voting_config.sheet_by_name(u'locations')

    location_data = []

    for i in range(location_sheet.nrows):
        location_data.append(
            location_sheet.row_values(
                i, start_colx=0, end_colx=None
            )
        )

    return location_data


if __name__ == '__main__':
    args = parser.parse_args()

    set_logging_level(logger, args.log)

    logger.info(f'reading {args.input_xlsx}')
    voting_config = xlrd.open_workbook(args.input_xlsx)

    # get voting locations from input xlsx file
    location_data = fetch_location_data(voting_config)

    voting_machines = pd.DataFrame(location_data)
    voting_machines.columns = voting_machines.iloc[0]
    voting_machines = voting_machines.iloc[1:]

    # sort voting location for optimization
    voting_machines.sort_values(['Likely or Exp. Voters', 'Eligible Voters',
                                'Ballot Length Measure'], ascending=False, inplace=True)

    # create location ID specific to new sort order
    voting_machines['Loc_ID'] = (voting_machines.index + 1).astype('int')
    voting_machines.head()

    # convert columns to numeric so they can be used for calculations
    voting_machine_nums = ['Likely or Exp. Voters',
                           'Eligible Voters', 'Ballot Length Measure']
    voting_machines[voting_machine_nums] = voting_machines[voting_machine_nums].astype('int')

    # convert ID to int
    voting_machines['ID'] = voting_machines['ID'].astype('int')

    voting_machines['Arrival_mean'] = \
        voting_machines['Likely or Exp. Voters'] / POLL_OPEN / 60

    voting_machines.head()

    # Create a dataframe to store location results

    num_machines = 50
    loc_df_results = create_loc_df(NUM_LOCATIONS + 1)

    for i in range(1, NUM_LOCATIONS):
        logger.info(f'Starting Location: {i}')

        # Placeholder, use a different start value for later machines?
        if (MIN_ALLOC_FLG):
            start_val = math.ceil((MAX_MACHINES - MIN_ALLOC) / 2) + MIN_ALLOC
            SASalphaValue = ALPHA_VALUE / math.log2(MAX_MACHINES - MIN_ALLOC)

            loc_res = izgbs(
                i,
                MAX_MACHINES,
                start_val,
                MIN_ALLOC,
                SASalphaValue
            )
        else:
            start_val = math.ceil((MAX_MACHINES - 1) / 2)
            SASalphaValue = ALPHA_VALUE / math.log2(MAX_MACHINES - MIN_ALLOC)
            loc_res = izgbs(
                i,
                num_machines,
                start_val,
                2,
                SASalphaValue
            )

        loc_feas = loc_res[loc_res['Feasible'] == 1].copy()

        if not loc_feas.empty:
            # calculate fewest feasible machines
            mach_min = loc_feas['Machines'].min()
            # Keep the feasible setup with the fewest number of machines
            loc_feas_min = loc_feas[loc_feas['Machines'] == mach_min].copy()
            # Populate overall results with info for this location
            # loc_df_results.loc[loc_df_results.Locations==i, 'Resource']==mach_min
            # loc_df_results.loc[loc_df_results.Locations==i, 'Exp. Avg. Wait Time']==loc_feas_min.iloc[0]['BatchAvg']
            # loc_df_results.loc[loc_df_results.Locations==i, 'Exp. Max. Wait Time']==loc_feas_min.iloc[0]['BatchAvgMax']
            loc_df_results.loc[
                loc_df_results.Locations == str(i), 'Resource'
            ] = mach_min

            loc_df_results.loc[
                loc_df_results.Locations == str(i), 'Exp. Avg. Wait Time'
            ] = loc_feas_min.iloc[0]['BatchAvg']

            loc_df_results.loc[
                loc_df_results.Locations == str(i), 'Exp. Max. Wait Time'
            ] = loc_feas_min.iloc[0]['BatchAvgMax']

            print(loc_df_results)
