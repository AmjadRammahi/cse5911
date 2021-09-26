# Not currently used
# This function uses indifference zone generalized binary search to find the number of resources for each location
def izgbs(NumberOfResources):
    Voters = voting_machines['Likely or Exp. Voters'].loc[(
        voting_machines.ID == vote_loc)][1]
    voters_max = voting_machines['Eligible Voters'].loc[(
        voting_machines.ID == vote_loc)][1]
    ballot_lth = voting_machines['Ballot Length Measure'].loc[(
        voting_machines.ID == vote_loc)][1]
    arrival_rt = voting_machines['Arrival_mean'].loc[(
        voting_machines.ID == vote_loc)][1]
    VotingMin, VotingMode, VotingMax, AverageVotingProcessing = voting_time_calcs(
        ballot_lth)
    vote_min, vote_mode, vote_max, vote_avg = voting_time_calcs(
        ballot_lengths[i])
    first_rep = 1

    # acceptable_sol=0
    # while (acceptable_sol==0):
    # for j in range (0, num_replications):
    # removed voters_est[i] from args
    # Start in the middle
    #num_test = (max_locs-1)/2
    #vote_min, vote_mode, vote_max, vote_avg = voting_time_calcs(ballot_lengths[i])
    # results_df = voter_sim(results_df, voters_max[i], vote_min, vote_mode, vote_max, period_start, period_finish, arrival_rate, num_test, loud=0)
    # Only keep results records that were actually used
    #results_df = results_df[results_df['Used']==1]
    # loc_sol[j] = num_test # replace with logic to test if it is the best solution
    #out_of_compliance_ct = 0
    # acceptable_sol=1