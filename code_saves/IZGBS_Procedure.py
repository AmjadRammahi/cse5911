def IZGBS_Procedure(machine_df, vote_loc, NumberOfResources):

    InternvalLValueUValue = 15
    replication = 1
    L0Value = math.ceil((Voters / (13 * 60)) / (1 / (AverageVotingProcessing)) + 0.5244 *
                        math.sqrt((Voters / (13 * 60)) / (1 / (AverageVotingProcessing))) + 0.5) - 2
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
    avgResources[i] = avgResources
    avgWaitingTime[i] = FinalAverageWaitingTime
    MaxWaitingTime[i] = FinalMaxWaitingTime

    if (Objective == "Quantile"):
        QuantWaitingTime[i] = FinalQuantileWaitingTime
        WaitProbabilities[i] = FinalProbabilityWait
        MeanClosingTimes[i] = FinalMeanPollClosingTime
