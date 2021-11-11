import numpy as np
from src.voter_sim import voter_sim
from src.settings import Settings

# partially done
def AKPIp1(n0InitialSample, k, waitingTime, sasAlphaValue, KnownMu0Value,
            deltaIZ, max_voters, expected_voters, vote_min, vote_mode, 
            vote_max, num_machines, avg_wait_time_avg, max_wait_time_avg):
    c = 1
    totalObservation = 0

    setI = np.empty((k + 1), int)
    sampleAverage = np.empty((k + 1), int)
    sampleVariance = np.empty((k), int)
    sample = np.empty([k+1, n0InitialSample], int)
    wFunction = np.empty((k), int)
    sumDeviation = np.empty((k), int)
    systemEliminated = [False] * k+1
    systemFeasible = [False] * k+1


    for i in range(0, k+1):
        setI[i] = i

    # calculate eta
    eta = ((2 - 2 * ((1 - sasAlphaValue) ** (1 / k))) ** (-2 / (n0InitialSample - 1)) - 1) / 2
   

    #if KnownMu0Value:
    iStartValue = 1 
    sampleAverage[0] = waitingTime
    for i in range(n0InitialSample):
        sample[0, i] = waitingTime
    # else:
    #     iStartValue = 0
    
    
    for i in range(iStartValue, k+1):
        sumNoise = 0
        sumNoiseAverage = 0
        sumNoiseMax = 0
        sumNoiseQuantile = 0
        for j in range(n0InitialSample):
            sample[i, j] =     # placeholder for voting sim
            
            sumNoiseAverage += avg_wait_time_avg
            sumNoiseMax += max_wait_time_avg
            #sumNoiseQuantile += batchedMeanQuantile
            sumNoise += sample[i,j]
        
        sampleAverage(i) = sumNoise / n0InitialSample
        AverageWaitingTimeForEvaluation = sumNoiseAverage / n0InitialSample
        MaxWaitingTimeForEvaluation = sumNoiseMax / n0InitialSample
        #QuantileWaitingTimeForEvaluation = sumNoiseQuantile / n0InitialSample

    for i in range(iStartValue, k+1):
        sumSquare = 0
        for j in range(n0InitialSample):
            sumSquare = sumSquare + (sample[i, j] - sampleAverage(i)) ** 2
        sampleVariance[i - 1] = sumSquare / (n0InitialSample - 1)

    r = n0InitialSample
    # Feasibility Check/Screening
    hsquare = 2 * c * eta * (n0InitialSample - 1)
    CountofRemainSystem = k + 1
    AlternativeMeanIsHigher = False
    
    while True:
        for i in range(CountofRemainSystem - 1):
            setII = setI[i+1]
            setII-= 1
            vwz = (hsquare * sampleVariance(i) / 2 / c / deltaIZ) - \
                (deltaIZ * r / 2 / c)

            if vwz >= 0:
                wFunction[setII] = vwz
            else:
                wFunction[setII] = 0
            
            sumDeviation[setII] = 0

            for j in range(r):
                sumDeviation[setII] = sumDeviation[setII] + sample[setII + 1, j] - waitingTime

            if sumDeviation[setII] <= -wFunction[setII]:
                systemFeasible[i] = True #change
            elif sumDeviation[setII] >= wFunction[setII]:
                systemEliminated[i] = True  #change
            
            NumberOfEliminatedSystem = 0

        for i in range(CountofRemainSystem):
                for j in range(i - NumberOfEliminatedSystem, \
                    CountofRemainSystem - 1 - NumberOfEliminatedSystem):
                    setI[j] = setI[j+1]
                    NumberOfEliminatedSystem += 1
                    if systemEliminated[i]:
                        AlternativeMeanIsHigher = True

        CountofRemainSystem -= NumberOfEliminatedSystem

            
        if CountofRemainSystem == 1:
            break
        else:
            r += 1
            # add more columns to sample array
            for j in range(CountofRemainSystem):
                setII = setI[i]
                if setII == 0 and KnownMu0Value:
                    sampleAverage[setII] = waitingTime
                    sampleAverage[setII, r] = waitingTime
                else:
                    sample[setII, r] = # placeholder for voter_sim
                    totalObservation += 1
                    sampleAverage[setII] = (sampleAverage[setII] * (r - 1) + sample[setII, r]) / r
                    AverageWaitingTimeForEvaluation = (AverageWaitingTimeForEvaluation * (r - 1) + batch_avg_wait_times) / r
                    MaxWaitingTimeForEvaluation = (MaxWaitingTimeForEvaluation * (r - 1) + batch_max_wait_times) / r
                    QuantileWaitingTimeForEvaluation = (QuantileWaitingTimeForEvaluation * (r - 1) + batchedMeanQuantile) / r

            for i in range(iStartValue, CountofRemainSystem):
                setII = setI[i]
                sumSquare = 0
                for j in range(r):
                    sumSquare = sumSquare + (sample[setII, j] - sampleAverage[setII])**2
                sampleVariance[setII] = sumSquare / (r - 1)


                
