def Single_BinarySearch(n0InitialSample, NumServers, alphaValue, deltaIZ, mu0Value, U0Value, L0Value, TotalVoters, EarlyVoterHours, PollHours, PeriodArrivalRate,
                        PeriodCompletion, Objective, TotalNoOfPeriods, VotingProcessingMin, VotingProcessingMode, VotingProcessingMax, MaxNumberOfVoters,
                        BatchSize, DisplayRow, FinalAverageWaitingTime, FinalMaxWaitingTime, FinalQuantileWaitingTime, InternvalLValueUValue, ObjectiveQuantileValue,
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