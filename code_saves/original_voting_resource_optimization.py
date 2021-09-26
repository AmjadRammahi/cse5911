# -*- coding: utf-8 -*-
"""Voter_Resource_Optimization.ipynb


"""

# Import all necessary packages
import numpy as np
#!pip install simpy
import simpy as sp
#!pip install multiprocessing
import multiprocessing as mp
import random as rand 
import pandas as pd
import math
import scipy.stats as st
import xlrd

# from google.colab import auth
# auth.authenticate_user()
import gspread
# from oauth2client.client import GoogleCredentials
# gc = gspread.authorize(GoogleCredentials.get_application_default())

print("Number of cpu : ", mp.cpu_count())

# Read in voting locations from Google link
# voting_config = gc.open_by_url('https://docs.google.com/spreadsheets/d/1Gq8DaVZw71RZ0lkE6PGmYD-DLAgcqTLZsjmjxZ_4Kdo/edit#gid=0')
# loc_sheet = voting_config.worksheet('Locations')
# loc_data = loc_sheet.get_all_values()

# Read in voting locations from Excel file
voting_config = xlrd.open_workbook('voting_excel.xlsx')
loc_sheet = voting_config.sheet_by_name(u'locations')
loc_data = list()
for i in range(loc_sheet.nrows):
  col_values = loc_sheet.row_values(i, start_colx=0, end_colx=None)
  loc_data.append(col_values)

voting_machines = pd.DataFrame(loc_data)
voting_machines.columns = voting_machines.iloc[0]
voting_machines = voting_machines.iloc[1:]
# Sort voting location for optimization
voting_machines.sort_values(['Likely or Exp. Voters', 'Eligible Voters', 'Ballot Length Measure'], ascending=False, inplace=True)
# Create location ID specific to new sort order
voting_machines['Loc_ID']=(voting_machines.index+1).astype('int')
voting_machines.head()

# Convert columns to numeric so they can be used for calculations
voting_machine_nums = ['Likely or Exp. Voters', 'Eligible Voters', 'Ballot Length Measure']
voting_machines[voting_machine_nums] = voting_machines[voting_machine_nums].astype('int')
# Convert ID to int
voting_machines['ID']=voting_machines['ID'].astype('int')

# Placeholder for input parameters
# Options: Apportionment, Allocation
prob_type = 'apportionment'
# If apportionment, this is the maximum total that can be purchased
# If allocation, this is the total number of machines available to be allocated
max_machines = 60

# General variables
# first and last voting locations. This allows results to be calculated for a subset of location
first_loc=1
last_loc=5
num_locations=last_loc-first_loc+1
# time increment in minutes
time_delta = 0.5 
# Probability of rejecting the null hypothesis
alphaValue = 0.05
# Indifference zone parameter
deltaIZ = 0.5
num_replications = 100

# deluxe version variable placeholders
# 5:30, 6:30, 7:30
early_start = 5.5
poll_start = 6.5
poll_end = 19.5
Poll_Hours = (poll_end-poll_start)*24
EarlyVoterHours = poll_start-early_start

# waiting time of voter who waits the longest
service_req = 30
    
# Batch size
BatchSize = 20   

# Specifies whether there is a minimum allocation requirement, and if so what it is
### Check that this is being used correctly
min_alloc_flg = 1
min_alloc=4
#min_alloc = 175

## Defining objective
# Current state: only "Max", future state to include "Quantile" and "Average"
Objective = 'Max'
# Not used in MVP
#ObjectiveQuantileValue = 0.95
# Waiting time <= how many minutes
mu0Value = 500

# Arrival time periods
## not currently used
#TotalNoOfPeriods = 4

# Voting times
MinVotingMin = 6    
MinVotingMode = 8
MinVotingMax = 12    
MinBallot = 0
MaxVotingMin = 6
MaxVotingMode = 10
MaxVotingMax = 20
MaxBallot = 10
# Only single resource has been built
NumberOfResources=1

# Not yet used
# check_in_times = [0.9, 1.1, 2.9]

# Calculate number of batches
num_batches = num_replications/BatchSize

# Add voter average arrivals in minutes
poll_open = poll_end-poll_start
voting_machines['Arrival_mean']=voting_machines['Likely or Exp. Voters']/poll_open/60
voting_machines.head()

# Create results arrays
avgResources = np.zeros(num_locations)
avgWaitingTime = np.zeros(num_locations)
MaxWaitingTime = np.zeros(num_locations)
QuantWaitingTime = np.zeros(num_locations)
WaitProbabilities = np.zeros(num_locations)
MeanClosingTimes = np.zeros(num_locations)

def voting_time_calcs(Ballot):
  votemin = MinVotingMin + (MaxVotingMin - MinVotingMin) / (MaxBallot - MinBallot) * (Ballot - MinBallot)
  votemode = MinVotingMode + (MaxVotingMode - MinVotingMode) / (MaxBallot - MinBallot) * (Ballot - MinBallot)
  votemax = MinVotingMax + (MaxVotingMax - MinVotingMax) / (MaxBallot - MinBallot) * (Ballot - MinBallot)
  voteavg = (votemin + votemode + votemax) / 3
  return votemin, votemode, votemax, voteavg

# Removed arguments: poll_open, poll_close, min_ballot_lth, max_ballot_lth, pred_voters, min_bal_length, max_bal_length,
# Loud parameter indicates whether voter progress should be printed to console
def voter_sim(res_df, max_voter, vote_time_min, vote_time_mode, vote_time_max, per_start, per_end, arrival_rt, num_machines, loud=0):
    #voter_waits = np.zeros(max_voter)
    #print(num_machines)
    RANDOM_SEED = 56 # repeatability during testing 
    SIM_TIME = (per_end - per_start) * 60    # Simulation time in minutes

    def vote_time():        
        # AverageVotingProcessing = (VotingProcessingMin + VotingProcessingMode + VotingProcessingMax) / 3
        return np.random.triangular(vote_time_min, vote_time_mode, vote_time_max, size=None)
    
    def generate_voter():
      return rand.expovariate(1.0/arrival_rt) 

    class Voting_Location(object):
        def __init__(self, env, num_machines):
            self.env = env
            self.machine = sp.Resource(env, capacity=num_machines)
            
        def vote(self, voter): 
            # randomly generate voting time 
            voting_time = vote_time()
            # voting_time = np.random.triangular(min_vote_time, mode_vote_time, max_vote_time, size=None)
            yield self.env.timeout(voting_time) 
            res_df.loc[res_df.Voter == voter, 'Voting_Time'] = voting_time
            if (loud==1): 
              print("%s voted in %d minutes." % (voter, voting_time))

    def voter(env, name, voter_num, vm):
        # Indicates that this row of the results df is used
        res_df.loc[res_df.Voter == name, 'Used'] = 1
        res_df.loc[res_df.Voter == name, 'Arrival_Time'] = env.now 
        if (loud == 1):
          print('%s arrives at the polls at %.2f.' % (name, env.now))
        with vm.machine.request() as request:
          yield request
          res_df.loc[res_df.Voter == name, 'Voting_Start_Time'] = env.now 
          if (loud==1):
            print('%s enters the polls at %.2f.' % (name, env.now))
          yield env.process(vm.vote(name))
          res_df.loc[res_df.Voter == name, 'Departure_Time'] = env.now 
          if (loud==1):
            print('%s leaves the polls at %.2f.' % (name, env.now))


    def setup(env, num_machines, arrival_rt, max_voters):
        # Create the voting machine
        voting_machine = Voting_Location(env, num_machines)
        
        # this logic prevents us from creating more than the specified maximum number of voters
        for i in range(max_voter): 
            # generate arrival time for next voter                        
            t = generate_voter()
            yield env.timeout(t)
            env.process(voter(env, 'Voter %d' %i, i, voting_machine)) 
            
    # Setup and start the simulation
    #print('Voting')
    # rand.seed(RANDOM_SEED)  # This helps reproduce the results
    # Create an environment and start the setup process
    env = sp.Environment()
    env.process(setup(env, num_machines, arrival_rt, max_voter))
    env.run(until=SIM_TIME) # environment will run until end of simulation time
    return res_df

# This function is used to create an empty dataframe to store simulation results
def create_res_df(max_voters):
  res_cols = [ "Replication", "Voter", "Arrival_Time", "Voting_Start_Time", "Voting_Time", "Departure_Time", "Waiting_Time", "Total_Time", "Used"]
  # Results dataframe with a record for the max possible number of voters
  voter_cols = np.zeros((max_voters, len(res_cols)))
  voter_results = pd.DataFrame(voter_cols, columns = res_cols)
  # Populates the voter ID field
  voter_results['Voter']= "Voter " + voter_results.index.astype('str')
  return voter_results

# This function creates a dataframe of IZGBS results to be outputted
def create_loc_df(vote_locs):
  res_cols = [ "Resource", "Exp. Avg. Wait Time", "Exp. Max. Wait Time"]
  # Create an empty dataframe the same size as the locations dataframe
  voter_cols = np.zeros((vote_locs, len(res_cols)))
  loc_results = pd.DataFrame(voter_cols, columns = res_cols)
  # Populates the location ID field
  loc_results['Locations'] = (loc_results.index+1).astype('str')
  return loc_results

# This function creates a dataframe to store hypothesis testing results
def create_hypothesis_df(num_h):
  # This dataframe will contain 1 record per machine count
  res_cols = [ "Machines", "Feasible", "BatchAvg", "BatchAvgMax"]
  # Create an empty dataframe the same size as the locations dataframe
  voter_cols = np.zeros((num_h, len(res_cols)))
  hyp_results = pd.DataFrame(voter_cols, columns = res_cols)
  # Populates the Machine count field
  hyp_results['Machines'] = (hyp_results.index+1).astype('int')
  return hyp_results

str(max_machines)

print(voting_machines['Likely or Exp. Voters'].loc[1])

voting_machines.head()

Voters = voting_machines['Likely or Exp. Voters'].loc[1]
Voters

def izgbs(vote_loc, total_num, start, min_val, alpha): 
# Read in parameters from locations dataframe
  # Voters = voting_machines['Likely or Exp. Voters'].loc[(voting_machines.Loc_ID==vote_loc)][0]
  # voters_max= voting_machines['Eligible Voters'].loc[(voting_machines.Loc_ID==vote_loc)][0]
  # ballot_lth = voting_machines['Ballot Length Measure'].loc[(voting_machines.Loc_ID==vote_loc)][0]
  # arrival_rt = voting_machines['Arrival_mean'].loc[(voting_machines.Loc_ID==vote_loc)][0]
  Voters = voting_machines['Likely or Exp. Voters'].loc[vote_loc]
  voters_max= voting_machines['Eligible Voters'].loc[vote_loc]
  ballot_lth = voting_machines['Ballot Length Measure'].loc[vote_loc]
  arrival_rt = voting_machines['Arrival_mean'].loc[vote_loc]

  # Calculate voting times
  vote_min, vote_mode, vote_max, vote_avg = voting_time_calcs(ballot_lth)

  # Create a dataframe for total number of machines
  feasible_df = create_hypothesis_df(total_num)

  # Start with the start value specified
  hyps_rem=1
  num_test = start
  cur_lower=min_val
  cur_upper=total_num
  while (hyps_rem):
    print("Num Test:" + str(num_test))
    print("Current Upper:" + str(cur_upper))
    print("Current Lower" + str(cur_lower))
    first_rep=1
    for i in range (1, num_replications):
      results_df = create_res_df(voters_max)
      # Calculate voting times   
      results_df = voter_sim(results_df, voters_max, vote_min, vote_mode, vote_max, poll_start, poll_end, arrival_rt, num_test, loud=0)
        # Only keep results records that were actually used
      results_df = results_df[results_df['Used']==1].copy()
      results_df['Replication']=i
      if (first_rep==1):
        final_loc_results = results_df.copy()
        first_rep=0
      else:
        final_loc_results = pd.concat([final_loc_results, results_df])
    # Calculate waiting time
    ##### Need to drop voters that did not finish
    final_loc_results['Waiting_Time']=final_loc_results['Voting_Start_Time']-final_loc_results['Arrival_Time']
    final_loc_results['Total_Time']=final_loc_results['Departure_Time']-final_loc_results['Arrival_Time']    
    
    # Calculate average and max waiting time by replication
    rep_nums = final_loc_results.groupby(['Replication'])['Waiting_Time'].agg(['max', 'mean']).reset_index()
    # Rename fields
    rep_nums.rename(columns={'max': 'Max_Waiting_Time', 'mean': 'Avg_Waiting_Time'}, inplace=True)
    # Assign records to batches
    rep_nums['Batch'] = rep_nums['Replication']%num_batches
    rep_nums.head()
    # Calculate batch averages
    calcs = ['Max_Waiting_Time', 'Avg_Waiting_Time']
    batch_avgs = rep_nums.groupby('Batch')[['Max_Waiting_Time', 'Avg_Waiting_Time']].agg('mean').reset_index()
    avg_avg=batch_avgs['Avg_Waiting_Time'].mean()
    max_avg=batch_avgs['Max_Waiting_Time'].mean()
    max_std=batch_avgs['Max_Waiting_Time'].std()
    batch_avgs.head()
    print(avg_avg, max_avg, max_std)
    # Populate results
    feasible_df.loc[feasible_df.Machines == num_test, 'BatchAvg'] = avg_avg
    feasible_df.loc[feasible_df.Machines == num_test, 'BatchAvgMax'] = max_avg
    # Calculate test statistic
    if (max_std > 0):
      z=(max_avg-(service_req+deltaIZ))/(max_std/math.sqrt(num_batches))
      p = st.norm.cdf(z)
      # feasible
      if (p < SASalphaValue):
        feasible_df.loc[feasible_df.Machines >= num_test, 'Feasible'] = 1
        print('scenario 1')
        # Move to lower half
        cur_upper = num_test
        t = math.floor((cur_upper-cur_lower)/2)
        num_test = math.floor((cur_upper-cur_lower)/2)+ cur_lower  
      else:
        print('scenario2')
        # Move to upper half
        feasible_df.loc[feasible_df.Machines == num_test, 'Feasible'] = 0
        cur_lower=num_test 
        num_test = math.floor((cur_upper-num_test)/2)+ cur_lower 
      # Declare feasible if Std = 0??
    else:
        print('scenario3')
        feasible_df.loc[feasible_df.Machines >= num_test, 'Feasible'] = 1
         # Move to lower half
        cur_upper = num_test
        t = math.floor((cur_upper-cur_lower)/2)
        num_test = math.floor((cur_upper-cur_lower)/2)+ cur_lower       
    # Check if there are hypotheses left to test
    # if (num_test >= min_val):
    #   hyps_rem=1
    # else:
    #   hyps_rem=0

    if (cur_upper > cur_lower and num_test > cur_lower and num_test < cur_upper):
      hyps_rem=1
    else:
      hyps_rem=0

  print(feasible_df)
  return feasible_df

num_locations

max_machines

# Create a dataframe to store location results
num_machines=50
loc_df_results = create_loc_df(num_locations+1)
print(loc_df_results)
for i in range(1, num_locations):
  print("Starting Location " + str(i))
  # Placeholder, use a different start value for later machines?
  if (min_alloc_flg):
    start_val = math.ceil((max_machines-min_alloc)/2)+min_alloc
    SASalphaValue = alphaValue / math.log2(max_machines-min_alloc)
    loc_res = izgbs(i, max_machines, start_val, min_alloc, SASalphaValue)
  else:
    start_val = math.ceil((max_machines-1)/2)
    SASalphaValue = alphaValue / math.log2(max_machines-min_alloc)
    loc_res = izgbs(i, num_machines, start_val, 2, SASalphaValue) 
  loc_feas = loc_res[loc_res['Feasible']==1].copy()
  if not loc_feas.empty:
    # calculate fewest feasible machines
    mach_min = loc_feas['Machines'].min()
    # Keep the feasible setup with the fewest number of machines
    loc_feas_min = loc_feas[loc_feas['Machines']==mach_min].copy()
    # Populate overall results with info for this location
    # loc_df_results.loc[loc_df_results.Locations==i, 'Resource']==mach_min
    # loc_df_results.loc[loc_df_results.Locations==i, 'Exp. Avg. Wait Time']==loc_feas_min.iloc[0]['BatchAvg']
    # loc_df_results.loc[loc_df_results.Locations==i, 'Exp. Max. Wait Time']==loc_feas_min.iloc[0]['BatchAvgMax'] 
    loc_df_results.loc[loc_df_results.Locations==str(i), 'Resource']=mach_min
    loc_df_results.loc[loc_df_results.Locations==str(i), 'Exp. Avg. Wait Time']=loc_feas_min.iloc[0]['BatchAvg']
    loc_df_results.loc[loc_df_results.Locations==str(i), 'Exp. Max. Wait Time']=loc_feas_min.iloc[0]['BatchAvgMax'] 
    # loc_df_results.head()
    print(loc_df_results)


#########################################################################3

##################### Not currently used
# This function uses indifference zone generalized binary search to find the number of resources for each location
def izgbs(NumberOfResources):
    Voters = voting_machines['Likely or Exp. Voters'].loc[(voting_machines.ID==vote_loc)][1]
    voters_max= voting_machines['Eligible Voters'].loc[(voting_machines.ID==vote_loc)][1]
    ballot_lth = voting_machines['Ballot Length Measure'].loc[(voting_machines.ID==vote_loc)][1]
    arrival_rt = voting_machines['Arrival_mean'].loc[(voting_machines.ID==vote_loc)][1]
    VotingMin, VotingMode, VotingMax, AverageVotingProcessing = voting_time_calcs(ballot_lth)
    vote_min, vote_mode, vote_max, vote_avg = voting_time_calcs(ballot_lengths[i])
    first_rep=1
   
  #acceptable_sol=0
  #while (acceptable_sol==0):
    #for j in range (0, num_replications):
        # removed voters_est[i] from args
        # Start in the middle
        #num_test = (max_locs-1)/2      
        #vote_min, vote_mode, vote_max, vote_avg = voting_time_calcs(ballot_lengths[i])
       # results_df = voter_sim(results_df, voters_max[i], vote_min, vote_mode, vote_max, period_start, period_finish, arrival_rate, num_test, loud=0)
        # Only keep results records that were actually used
        #results_df = results_df[results_df['Used']==1]
        #loc_sol[j] = num_test # replace with logic to test if it is the best solution
        #out_of_compliance_ct = 0
    #acceptable_sol=1

# machine_df contains estimated voters, max voters, and ballot length
def IZGBS_Procedure(machine_df, vote_loc, NumberOfResources):

  InternvalLValueUValue = 15
  replication = 1
  L0Value = math.ceil((Voters / (13 * 60)) / (1 / (AverageVotingProcessing)) + 0.5244 * math.sqrt((Voters / (13 * 60)) / (1 / (AverageVotingProcessing))) + 0.5) - 2
  L0Value = max(L0Value, 3)
  U0Value = L0Value + InternvalLValueUValue
  sum_avgResources = 0

  if (min_alloc_flg == 1):
    L0Value = max(L0Value, min_alloc)

        
  for n_rep in range(1, replication):
    # Call Single_BinarySearch function
    Single_BinarySearch(n0InitialSample, xValue, alphaValue, deltaIZ, mu0Value, 
                        U0Value, L0Value, Voters, EarlyVoterHours, PollHours, PeriodArrivalRate, 
                        PeriodCompletion, Objective, TotalNoOfPeriods, VotingProcessingMin,
                        VotingProcessingMode, VotingProcessingMax, MaxNumberOfVoters, BatchSize,
                        DisplayRow, FinalAverageWaitingTime, FinalMaxWaitingTime, FinalQuantileWaitingTime, 
                        InternvalLValueUValue, ObjectiveQuantileValue, NumberOfResources, 
                        FinalProbabilityWait, FinalMeanPollClosingTime)
    sum_avgResources = sum_avgResources + xValue
    Resource[n_rep] = xValue

  avgResources = sum_avgResources / replication
  # Populate results
  avgResources[i]=avgResources
  avgWaitingTime[i]=FinalAverageWaitingTime
  MaxWaitingTime[i]=FinalMaxWaitingTime
        
  if (Objective == "Quantile"): 
    QuantWaitingTime[i] = FinalQuantileWaitingTime
    WaitProbabilities[i] = FinalProbabilityWait
    MeanClosingTimes[i] =  FinalMeanPollClosingTime

# placeholder for algorithm to determine next configuration to test. This should be moved to location loop.
num_test = 2
# Iterate over locations
loc_sol = np.zeros(num_locations) # number machines
loc_waits = np.zeros(num_locations) # voter wait times
loc_ct = np.zeros(num_locations) # number of out of compliance voters at each location, not currently being used

# for i in range(first_loc, last_loc+1):
#     # Create a results df

#     # Pull simulation parameters    
#     #IZGBS_Procedure(voting_machines, i, NumberOfResources)
#     for i in range(1:replication):
#         # logic not currently used because individual times are not captured yet
#         # for k in loc_waits[i]:
#             # if k > service_req + time_delta:
#                 # out_of_compliance_ct = out_of_compliance_ct + 1
#     #loc_ct[j] = out_of_compliance_ct
# # placeholder for logic to write to a file
# # f = open('sim_results_test','w')
# # for k in range(0, loc_sol - 1):
#     # f.write('location ' + k + ': number machines ' + loc_sol[k] + '\n')
#     # f.write(loc_waits + '\n')
#     # f.write('location ' + k + ': out of compliance count ' + loc_ct[k] + '\n')
# # f.close()

def AKPIp1(n0InitialSample, alphaValue, deltaIZ, AlternativeMeanIsHigher, k, mux, 
           variance, KnownMu0Value, TotalObservation, NumServers, NumServers_Station2, TotalVoters,
           BatchSize, EarlyVoterHours, PollHours, PeriodArrivalRate, PeriodCompletion, Objective, 
           TotalNoOfPeriods, MaxNumberOfVoters, RegistrationProcessingMin, RegistrationProcessingMode, 
           RegistrationProcessingMax, VotingProcessingMin, VotingProcessingMode, VotingProcessingMax, 
           ObjectiveQuantileValue, NumberOfResources, AverageWaitingTimeForEvaluation, MaxWaitingTimeForEvaluation, 
           QuantileWaitingTimeForEvaluation, batchedMeanProbabilityWait, batchedMeanPollClosingTime):
# Shijie Huang(8/2016): The programming in this sub is coded based on Phase I of a constrained R&S procedure (Andradottir and Kim 2010). The objective is to identify and return a set consisting of all the desirable systems, possibly including some acceptable systems.
# Andradottir, S., Kim, S. H. 2010. Fully sequential procedures for comparing constrained systems via simulation. Naval Research Logistics (NRL) 57(5) 403-421.

# Step 0.Setup
  TotalObservation = 0
  # Step 0a. Select a positive integer
  c = 1

  # Step 0b. Set Set I
  SetI = np.zeros(k) 
  for i in range(0,k):
    SetI[i] = i

  # Step 0c. Calculate eta
  eta = ((2 - 2 * ((1 - alphaValue) ^ (1 / k))) ^ (-2 / (n0InitialSample - 1)) - 1) / 2

# Step 1.Initialization

  # Step 1a. Calculate sample average and std from n0 observations
  if (KnownMu0Value == True):
    IStartValue = 1
    sampleAverage[0] = mux[0]
    for j in range(1, n0InitialSample):
      # Verify that this line is working correctly
      Sample[0, j] = mux[0]
  else:
    IStartValue = 0

  for i in range(IStartValue, k):
    sumNoise = 0
    sumNoiseAverage = 0
    sumNoiseMax = 0
    sumNoiseQuantile = 0
    for j in range(1, n0InitialSample):
      # Placeholder for simulation call
      #Sample(i, j) = TandemQueueWQuartile()
      sumNoiseAverage = sumNoiseAverage + batchedMeanMean 
      sumNoiseMax = sumNoiseMax + batchedMeanMax
      sumNoiseQuantile = sumNoiseQuantile + batchedMeanQuantile
        
    TotalObservation = TotalObservation + 1
    sumNoise = sumNoise + Sample(i, j)
    SampleAverage[i] = sumNoise / n0InitialSample
    AverageWaitingTimeForEvaluation = sumNoiseAverage / n0InitialSample
    MaxWaitingTimeForEvaluation = sumNoiseMax / n0InitialSample
    QuantileWaitingTimeForEvaluation = sumNoiseQuantile / n0InitialSample

  for i in range(IStartValue, k):
    sumSquare = 0
    for j in range(1, n0InitialSample):
        sumSquare = sumSquare + (Sample(i, j) - sampleAverage(i)) ^ 2
        SampleVariance[i] = sumSquare / (n0InitialSample - 1)

  # Step 1b. Set stage counter
  r = n0InitialSample

# Step 2. Feasibility Check/Screening
  hsquare = 2 * c * eta * (n0InitialSample - 1)
  CountofRemainSystem = k + 1
  AlternativeMeanIsHigher = False

  for i in range(1, CountofRemainSystem - 1):
    SetII = SetI[i]
    vwz = (hsquare * SampleVariance[i] / 2 / c / deltaIZ) - (deltaIZ * r / 2 / c)
    if vwz >= 0:
      WFunction[SetII] = vwz
    else:
      WFunction[SetII] = 0
      SumDeviation[SetII] = 0    
    for j in range(1, r+1): # check indexing
      SumDeviation[SetII] = SumDeviation[SetII] + Sample[SetII, j] - mux(0)

    if SumDeviation[SetII] <= -WFunction[SetII]:
      SystemFeasible[i] = True
    elif SumDeviation[SetII] >= WFunction[SetII]:
            SystemEliminated[i] = True
        
   
    NumberOfEliminatedSystem = 0
  for i in range(0, CountofRemainSystem):
    if SystemFeasible[i] == True:
      for j in range(i - NumberOfEliminatedSystem, CountofRemainSystem - 1 - NumberOfEliminatedSystem):
          SetI[j] = SetI[j + 1]
          NumberOfEliminatedSystem = NumberOfEliminatedSystem + 1
    elif SystemEliminated(i) == True:
      for j in range(i - NumberOfEliminatedSystem, CountofRemainSystem - 1 - NumberOfEliminatedSystem):
        SetI[j] = SetI(j + 1)
        NumberOfEliminatedSystem = NumberOfEliminatedSystem + 1
        AlternativeMeanIsHigher = True # This statement is only good for one alterative system

    
  if NumberOfEliminatedSystem > 0:
    CountofRemainSystem = CountofRemainSystem - NumberOfEliminatedSystem

  if CountofRemainSystem != 1:
    r = r + 1
    for j in range (0, CountofRemainSystem):
      SetII = SetI[j]
      if SetII == 0 and KnownMu0Value == True:
        sampleAverage[SetII] = mux(0)
        Sample[SetII, r] = mux(0)
      else:
        Sample[SetII, r] = TandemQueueWQuartile(NumServers, NumServers_Station2, TotalVoters, EarlyVoterHours, 
                                                PollHours, PeriodArrivalRate, PeriodCompletion, Objective, 
                                                TotalNoOfPeriods, MaxNumberOfVoters, BatchSize, RegistrationProcessingMin, 
                                                RegistrationProcessingMode, RegistrationProcessingMax, VotingProcessingMin,
                                                VotingProcessingMode, VotingProcessingMax, batchedMeanQuantile, batchedMeanMax, batchedMeanMean,
                                                ObjectiveQuantileValue, NumberOfResources, batchedMeanProbabilityWait, batchedMeanPollClosingTime)
        TotalObservation = TotalObservation + 1
        sampleAverage[SetII] = (sampleAverage[SetII] * (r - 1) + Sample[SetII, r]) / r
        AverageWaitingTimeForEvaluation = (AverageWaitingTimeForEvaluation * (r - 1) + batchedMeanMean) / r
        MaxWaitingTimeForEvaluation = (MaxWaitingTimeForEvaluation * (r - 1) + batchedMeanMax) / r
        QuantileWaitingTimeForEvaluation = (QuantileWaitingTimeForEvaluation * (r - 1) + batchedMeanQuantile) / r
            
   
        
    for i in range(IStartValue, CountofRemainSystem):
      SetII = SetI[i]
      sumSquare = 0
      # Check this indexing
      for j in range(1, r+1):
        sumSquare = sumSquare + (Sample[SetII, j] - sampleAverage[SetII]) ^ 2

def Single_BinarySearch(n0InitialSample, NumServers, alphaValue, deltaIZ, mu0Value, U0Value, L0Value, TotalVoters, EarlyVoterHours, PollHours, PeriodArrivalRate,
                        PeriodCompletion, Objective, TotalNoOfPeriods, VotingProcessingMin, VotingProcessingMode, VotingProcessingMax, MaxNumberOfVoters,
                        BatchSize, DisplayRow , FinalAverageWaitingTime, FinalMaxWaitingTime, FinalQuantileWaitingTime, InternvalLValueUValue, ObjectiveQuantileValue, 
                        NumberOfResources, FinalProbabilityWait, FinalMeanPollClosingTime):
# The programming in this sub is coded for the binary search method for single resource

  NumServers_Station2 = 0
  UValue = U0Value
  LValue = L0Value
  if (U0Value - L0Value > 2):
    SASalphaValue = alphaValue / math.log(U0Value - L0Value, 2)
  else:
    SASalphaValue = alphaValue

  sum_n = 0
  NumofSystem = 1
  KnownMu0Value = True

  AKPImux[0] = mu0Value
  variance[0] = 0

  Flag = False
  while (UValue > LValue + 1):
    NumServers = math.ceil((LValue + UValue) / 2, 0)
  # Call AKPIp1 function
    meanIsHigher = AKPIp1(n0InitialSample, SASalphaValue, deltaIZ, meanIsHigher, NumofSystem, AKPImux, variance, 
      KnownMu0Value, nValue, NumServers, NumServers_Station2, TotalVoters, BatchSize, EarlyVoterHours, 
      PollHours, PeriodArrivalRate, PeriodCompletion, Objective, TotalNoOfPeriods, 
      MaxNumberOfVoters, RegistrationProcessingMin, RegistrationProcessingMode, 
      RegistrationProcessingMax, VotingProcessingMin, VotingProcessingMode, VotingProcessingMax, 
      ObjectiveQuantileValue, NumberOfResources, AverageWaitingTimeForEvaluation, 
      MaxWaitingTimeForEvaluation, QuantileWaitingTimeForEvaluation, 
      batchedMeanProbabilityWait, batchedMeanPollClosingTime)
     
    if (meanIsHigher == True):
      LValue = NumServers
    else:
      UValue = NumServers
      FinalAverageWaitingTime = AverageWaitingTimeForEvaluation
      FinalMaxWaitingTime = MaxWaitingTimeForEvaluation
      FinalQuantileWaitingTime = QuantileWaitingTimeForEvaluation
      FinalProbabilityWait = batchedMeanProbabilityWait
      FinalMeanPollClosingTime = batchedMeanPollClosingTime / 60
      Flag = True
      sum_n = sum_n + nValue

    if (UValue == U0Value): 
      U0Value = U0Value + InternvalLValueUValue - 2
      L0Value = L0Value + InternvalLValueUValue - 2
      UValue = U0Value
      LValue = L0Value

    if (LValue == L0Value and LValue >= 2):
      L0Value = L0Value - min(L0Value - 1, InternvalLValueUValue - 2)
      U0Value = U0Value - min(L0Value - 1, InternvalLValueUValue - 2)
      UValue = U0Value
      LValue = L0Value

  NumServers = UValue

  if (Flag == False):
    AKPIp1(n0InitialSample, SASalphaValue, deltaIZ, meanIsHigher, NumofSystem, 
    AKPImux, variance, KnownMu0Value, nValue, NumServers, NumServers_Station2, 
    TotalVoters, BatchSize, EarlyVoterHours, PollHours, PeriodArrivalRate, 
    PeriodCompletion, Objective, TotalNoOfPeriods, MaxNumberOfVoters, 
    RegistrationProcessingMin, RegistrationProcessingMode, RegistrationProcessingMax, 
    VotingProcessingMin, VotingProcessingMode, VotingProcessingMax, ObjectiveQuantileValue, 
    NumberOfResources, AverageWaitingTimeForEvaluation, MaxWaitingTimeForEvaluation, 
    QuantileWaitingTimeForEvaluation, batchedMeanProbabilityWait, batchedMeanPollClosingTime)
     
    FinalAverageWaitingTime = AverageWaitingTimeForEvaluation
    FinalMaxWaitingTime = MaxWaitingTimeForEvaluation
    FinalQuantileWaitingTime = QuantileWaitingTimeForEvaluation
    FinalProbabilityWait = batchedMeanProbabilityWait
    FinalMeanPollClosingTime = batchedMeanPollClosingTime / 60

# Multiprocessing
# Call binary search for locations in parallel
