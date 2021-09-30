# Placeholder for input parameters
# Options: Apportionment, Allocation
prob_type = 'apportionment'  # NOTE: unused
# If apportionment, this is the maximum total that can be purchased
# If allocation, this is the total number of machines available to be allocated


time_delta = 0.5  # NOTE: unused - time increment in minutes


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


# Only single resource has been built
NumberOfResources = 1  # NOTE: unused


# Create results arrays
avgResources = np.zeros(NUM_LOCATIONS)  # NOTE: unused
avgWaitingTime = np.zeros(NUM_LOCATIONS)  # NOTE: unused
MaxWaitingTime = np.zeros(NUM_LOCATIONS)  # NOTE: unused
QuantWaitingTime = np.zeros(NUM_LOCATIONS)  # NOTE: unused
WaitProbabilities = np.zeros(NUM_LOCATIONS)  # NOTE: unused
MeanClosingTimes = np.zeros(NUM_LOCATIONS)  # NOTE: unused


# Iterate over locations
loc_sol = np.zeros(NUM_LOCATIONS)  # NOTE: unused - number machines
loc_waits = np.zeros(NUM_LOCATIONS)  # NOTE: unused - voter wait times
loc_ct = np.zeros(NUM_LOCATIONS)  # NOTE: unused


EARLY_START = 5.5  # NOTE: effectively unused
Poll_Hours = (POLL_END - POLL_START) * 24  # NOTE: unused
EarlyVoterHours = POLL_START - EARLY_START  # NOTE: unused
