import math
import logging
import numpy as np
import scipy.stats as st
from statistics import mean
import src.global_var
from src.voter_sim import voter_sim

'''
    get h using eta
    R - might be rinoaughts constant 
'''

def AKPIp1(n0InitialSample, alphaValue, deltaIZ, AlternativeMeanIsHigher, k, mux,
           variance, KnownMu0Value, TotalObservation, NumServers, NumServers_Station2, TotalVoters,
           BatchSize, EarlyVoterHours, PollHours, PeriodArrivalRate, PeriodCompletion, Objective,
           TotalNoOfPeriods, MaxNumberOfVoters, RegistrationProcessingMin, RegistrationProcessingMode,
           RegistrationProcessingMax, VotingProcessingMin, VotingProcessingMode, VotingProcessingMax,
           ObjectiveQuantileValue, NumberOfResources, AverageWaitingTimeForEvaluation, MaxWaitingTimeForEvaluation,
           QuantileWaitingTimeForEvaluation, batchedMeanProbabilityWait, batchedMeanPollClosingTime):

    '''
        params:
            n0InitialSample = 20
            alphaValue = sas_alpha_value
            deltaIZ = ALPHA_VALUE = 0.5 from source global
            KnownMu0Value always true for apportionment 
            k = 1  = number of systems
            c = 1
            r = 1   (number of systems to test)
            AlternativeMeanIsHigher = boolean value
            mux = waiting time
            TotalObservation start at zero
            NumServers : max number of service requirement in golbal var
            NumServers_Station2 = 0
            TotalVoters: likely to vote?
            BatchSize : MAX_VOTING_MAX 
            EarlyVoterHours in global var
            PollHours in global var
            PeriodArrivalRate : total of voters / poll hours open / 60
            PeriodArrivalRate : exp voters / SumOfArrivalRateTimesHours / 60 * arrival rate ratio (Home)
            (SumOfArrivalRateTimesHours = SumOfArrivalRateTimesHours + Arraival Ratio (Home) * period finish (Home) - period start (Home))
            PeriodCompletion = Period finish - Poll start in gloval var * 24 * 60
            Objective: Max, Average, Quantile
            TotalNoOfPeriods: arrivals patterns
            MaxNumberOfVoters: Eligible voters
            RegistrationProcessingMin: check in min (Home)
            RegistrationProcessingMode: check in mode (Home)
            RegistrationProcessingMax:	checkin max (Home)
            VotingProcessingMin: Setting 
            VotingProcessingMode: Setting
            VotingProcessingMaxL Setting 
            ObjectiveQuantileValue: Max, Average, Quantile
            NumberOfResources: int
            AverageWaitingTimeForEvaluation: sumNoiseAverage(start at 0) / n0InitialSample(20)
            MaxWaitingTimeForEvaluation: sumNoiseMax (start at 0)/ n0InitialSample (20)
            QuantileWaitingTimeForEvaluation = sumNoiseQuantile (start at 0)/ n0InitialSample(20)
            sumNoiseQuantile = sumNoiseQuantile + batchedMeanQuantile
            batchedMeanProbabilityWait: ???
            batchedMeanPollClosingTime: ???
    '''
    
    # Step 0.Setup
    TotalObservation = 0
    # Step 0a. Select a positive integer
    c = 1
    # Step 0b. Set Set I
    SetI = np.zeros(k)

    for i in range(0, k):
        SetI[i] = i

    # Step 0c. Calculate eta
    eta = ((2 - 2 * ((1 - alphaValue) ^ (1 / k)))^(-2 / (n0InitialSample - 1)) - 1) / 2
    
    # Step 1.Initialization
    
    # Step 1a. Calculate sample average and std from n0 observations
    if (KnownMu0Value == True):
        IStartValue = 1
        sampleAverage[0] = mux[0]
        for j in range(1, n0InitialSample):
            # Verify that this line is working correctly
            # change to Sample = [mux for i in range(1, n0InitialSample)]
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
            # TandemQueueWQuartile() vs voter_sim() ??
            # Sample(i, j) = TandemQueueWQuartile()
            sumNoiseAverage = sumNoiseAverage + batchedMeanMean # change to 
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
        vwz = (hsquare * SampleVariance[i] / 2 /
               c / deltaIZ) - (deltaIZ * r / 2 / c)
        if vwz >= 0:
            WFunction[SetII] = vwz
        else:
            WFunction[SetII] = 0
            SumDeviation[SetII] = 0
        for j in range(1, r+1):  # check indexing
            SumDeviation[SetII] = SumDeviation[SetII] + \
                Sample[SetII, j] - mux(0)

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
                # This statement is only good for one alterative system
                AlternativeMeanIsHigher = True

    if NumberOfEliminatedSystem > 0:
        CountofRemainSystem = CountofRemainSystem - NumberOfEliminatedSystem

    if CountofRemainSystem != 1:
        r = r + 1
        for j in range(0, CountofRemainSystem):
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
                sampleAverage[SetII] = (
                    sampleAverage[SetII] * (r - 1) + Sample[SetII, r]) / r
                AverageWaitingTimeForEvaluation = (
                    AverageWaitingTimeForEvaluation * (r - 1) + batchedMeanMean) / r
                MaxWaitingTimeForEvaluation = (
                    MaxWaitingTimeForEvaluation * (r - 1) + batchedMeanMax) / r
                QuantileWaitingTimeForEvaluation = (
                    QuantileWaitingTimeForEvaluation * (r - 1) + batchedMeanQuantile) / r

        for i in range(IStartValue, CountofRemainSystem):
            SetII = SetI[i]
            sumSquare = 0
            # Check this indexing
            for j in range(1, r+1):
                sumSquare = sumSquare + \
                    (Sample[SetII, j] - sampleAverage[SetII]) ^ 2