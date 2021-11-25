from typing import Callable, Generator, Iterable, Optional
import numpy as np
from scipy.stats import beta

from src.voter_sim import voter_sim


def presort_HD_L(wait_times: list, q: float) -> float:
    '''
        # TODO
    '''
    s = 0
    length = len(wait_times)

    a = q * (length + 1)
    b = (1 - q) * (length + 1)

    for i in range(length):
        s += (beta.pdf((i + 1) / length, a, b) - beta.pdf(i / length, a, b)) * wait_times[i]

    return s


def sum_and_max(wait_times: list) -> tuple:
    '''
        # TODO
    '''
    s, m = 0, 0

    for w in wait_times:
        s += w
        m = max(m, w)

    return s, m


def run_voter_sim(
    *,
    max_voters,
    expected_voters,
    vote_time_min,
    vote_time_mode,
    vote_time_max,
    num_machines,
    settings: dict
):
    '''
        # TODO
    '''
    batched_mean_mean = 0
    batched_mean_max = 0
    batched_mean_quantile = 0
    # replicate_quantile = [0] * settings['BATCH_SIZE']

    for _ in range(settings['BATCH_SIZE']):
        wait_times = voter_sim(
            max_voters=max_voters,
            expected_voters=expected_voters,
            vote_time_min=vote_time_min,
            vote_time_mode=vote_time_mode,
            vote_time_max=vote_time_max,
            num_machines=num_machines,
            settings=settings
        )

        s, m = sum_and_max(wait_times)

        replicate_avg_wait = s / len(wait_times)
        replicate_max_wait = m
        # replicate_quantile = presort_HD_L(wait_times, settings['OBJECTIVE_QUANTILE_VALUE'])

        batched_mean_mean += replicate_avg_wait
        batched_mean_max += replicate_max_wait
        # batched_mean_quantile += replicate_quantile

    batched_mean_mean /= settings['BATCH_SIZE']
    batched_mean_max /= settings['BATCH_SIZE']
    batched_mean_quantile /= settings['BATCH_SIZE']

    return batched_mean_mean, batched_mean_max  # , batched_mean_quantile


# NOTE: as far as I can tell, the VBA lists are 1-indexed so I am padding the lists to be size + 1
def redim(size: int, _type: Callable, *, preserve: Optional[list] = None) -> list:
    if preserve is not None:
        return preserve[:size + 1] \
            if len(preserve) > size + 1 \
            else preserve + [_type()] * (size + 1 - len(preserve))
    else:
        return [_type() for _ in range(size + 1)]


def vba_range(start: int, stop: int) -> Generator:
    yield from range(start, stop + 1)


def AKPIp1(
    *,
    sas_alpha_value,

    max_voters,
    expected_voters,
    vote_min,
    vote_mode,
    vote_max,
    num_machines,
    settings: dict
):
    '''
        # TODO
    '''
    # Step 0. Setup

    initial_sample = settings['INITIAL_SAMPLE_SIZE']
    wait_time = settings['SERVICE_REQ']
    delta_IZ = settings['DELTA_INDIFFERENCE_ZONE']
    k = 1

    sample_average = redim(k, float)
    sample_variance = redim(k, float)
    sample = redim(initial_sample, float)  # NOTE: this is the only 2D list, k=1 always so it can be reduced to 1D

    # Step 0a. Select a positive integer
    c = 1
    totalObservation = 0

    # Step 0b. Set Set I
    setI = redim(k, int)

    # Step 0c. Calculate eta
    eta = ((2 - 2 * ((1 - sas_alpha_value) ** (1 / k))) ** (-2 / (initial_sample - 1)) - 1) / 2

    # Step 1. Initialization

    # Step 1a. Calculate sample average and std from n0 observations

    # NOTE: KnownMu0Value = True - is a tautology, removed it
    I_start_value = 1
    sample_average[I_start_value] = wait_time

    for j in vba_range(1, initial_sample):
        sample[j] = wait_time

    for i in vba_range(I_start_value, k):
        sumNoise = 0
        sumNoiseAverage = 0
        sumNoiseMax = 0
        sumNoiseQuantile = 0

        for j in vba_range(1, initial_sample):
            # batched_mean_mean, batched_mean_max, batched_mean_quantile = run_voter_sim(
            batched_mean_mean, batched_mean_max = run_voter_sim(
                max_voters=max_voters,
                expected_voters=expected_voters,
                vote_time_min=vote_min,
                vote_time_mode=vote_mode,
                vote_time_max=vote_max,
                num_machines=num_machines,
                settings=settings
            )

            sample[j] = batched_mean_max  # NOTE: this was originally decided by Objective (in Tandem), ours is always Max

            sumNoiseAverage += batched_mean_mean
            sumNoiseMax += batched_mean_max
            #  sumNoiseQuantile += batched_mean_quantile

            totalObservation += 1
            sumNoise += sample[j]

        sample_average[i] = sumNoise / initial_sample
        Averagewait_timeForEvaluation = sumNoiseAverage / initial_sample
        Maxwait_timeForEvaluation = sumNoiseMax / initial_sample
        #  Quantilewait_timeForEvaluation = sumNoiseQuantile / initial_sample

    for i in vba_range(I_start_value, k):
        sumSquare = 0

        for j in vba_range(1, initial_sample):
            sumSquare = sumSquare + (sample[j] - sample_average[i]) ** 2

        sample_variance[i] = sumSquare / (initial_sample - 1)

    # Step 1b. Set stage counter

    r = initial_sample

    # Step 2. Feasibility Check/Screening
    hsquare = 2 * c * eta * (initial_sample - 1)

    WFunction = redim(k, float)
    SumDeviation = redim(k, float)
    SystemEliminated = redim(k, bool)
    SystemFeasible = redim(k, bool)

    CountofRemainSystem = k + 1
    mean_is_higher = False

    while True:
        for i in vba_range(1, CountofRemainSystem - 1):
            setII = setI[i]
            vwz = (hsquare * sample_variance[i] / 2 / c / delta_IZ) - (delta_IZ * r / 2 / c)

            if vwz >= 0:
                WFunction[setII] = vwz
            else:
                WFunction[setII] = 0

            SumDeviation[setII] = 0

            for j in vba_range(1, r):
                SumDeviation[setII] += sample[j] - wait_time

            if SumDeviation[setII] <= -WFunction[setII]:
                SystemFeasible[i] = True
            elif SumDeviation[setII] >= WFunction[setII]:
                SystemEliminated[i] = True

        NumberOfEliminatedSystem = 0

        for i in vba_range(0, CountofRemainSystem - 1):
            if SystemFeasible[i] is True:
                for j in vba_range(i - NumberOfEliminatedSystem, CountofRemainSystem - 2 - NumberOfEliminatedSystem):
                    setI[j] = setI[j + 1]

                NumberOfEliminatedSystem += 1
            elif SystemEliminated[i] is True:
                for j in vba_range(i - NumberOfEliminatedSystem, CountofRemainSystem - 2 - NumberOfEliminatedSystem):
                    setI[j] = setI[j + 1]

                NumberOfEliminatedSystem += 1
                mean_is_higher = True

        if NumberOfEliminatedSystem > 0:
            CountofRemainSystem -= NumberOfEliminatedSystem

            setI = redim(CountofRemainSystem - 1, int, preserve=setI)
            SystemEliminated = redim(CountofRemainSystem - 1, bool)
            SystemFeasible = redim(CountofRemainSystem - 1, bool)

        if CountofRemainSystem == 1:
            break
        else:
            r += 1
            sample = redim(r, float, preserve=sample)

            for j in vba_range(0, CountofRemainSystem - 1):
                setII = setI[j]

                if setII == 0:  # NOTE: removed - KnownMu0Value = True
                    sample_average[setII] = wait_time
                    sample[r] = wait_time
                else:
                    # batched_mean_mean, batched_mean_max, batched_mean_quantile = run_voter_sim(
                    batched_mean_mean, batched_mean_max = run_voter_sim(
                        max_voters=max_voters,
                        expected_voters=expected_voters,
                        vote_time_min=vote_min,
                        vote_time_mode=vote_mode,
                        vote_time_max=vote_max,
                        num_machines=num_machines,
                        settings=settings
                    )

                    sample[r] = batched_mean_max

                    totalObservation += 1
                    sample_average[setII] = (sample_average[setII] * (r - 1) + sample[r]) / r

                    Averagewait_timeForEvaluation = (Averagewait_timeForEvaluation * (r - 1) + batched_mean_mean) / r
                    Maxwait_timeForEvaluation = (Maxwait_timeForEvaluation * (r - 1) + batched_mean_max) / r
                    # Quantilewait_timeForEvaluation = (
                    #     Quantilewait_timeForEvaluation * (r - 1) + batched_mean_quantile) / r

            for i in vba_range(I_start_value, CountofRemainSystem - 1):
                setII = setI[i]
                sumSquare = 0

                for j in vba_range(1, r):
                    sumSquare += (sample[j] - sample_average[setII]) ** 2

                sample_variance[setII] = sumSquare / (r - 1)

    return mean_is_higher, Averagewait_timeForEvaluation, Maxwait_timeForEvaluation  # , Quantilewait_timeForEvaluation
