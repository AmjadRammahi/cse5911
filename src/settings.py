# Project Globals

# NOTE: this class holds all of the Settings needed to run this codebase.
# Do not create an instance of this class. Instead, if you need to modify
# a setting, then directly edit the class variable, ex: Settings.MIN_ALLOC_FLG = False
# from numba import int32, float32
# from numba.experimental import jitclass

# spec = [
#     ('POLL_START', float32[:]),
#     ('POLL_END', float32[:]),
#     ('POLL_OPEN', float32[:]),
#     ('BATCH_SIZE', int32),
#     ('NUM_REPLICATIONS', int32),
#     ('NUM_BATCHES', float32),
#     ('MIN_VOTING_MIN', int32),
#     ('MIN_VOTING_MODE', int32),
#     ('MIN_VOTING_MAX', int32),
#     ('MIN_BALLOT', int32),
#     ('MAX_VOTING_MIN', int32),
#     ('MAX_VOTING_MODE', int32),
#     ('MAX_VOTING_MAX', int32),
#     ('MAX_BALLOT', int32),
#     ('SERVICE_REQ', float32[:]),
#     ('MAX_MACHINES', int32),
#     ('ALPHA_VALUE', float32[:]),
#     ('DELTA_INDIFFERENCE_ZONE', float32[:]),
#     ('MIN_ALLOC_FLG', int32),
#     ('MIN_ALLOC', int32),
#     ('NUM_LOCATIONS', int32)
# ]


class Settings:
    POLL_START = 6.5  # 6:30 am
    POLL_END = 19.5  # 7:30 pm
    POLL_OPEN = POLL_END - POLL_START

    BATCH_SIZE = 4
    NUM_REPLICATIONS = 20
    NUM_BATCHES = NUM_REPLICATIONS // BATCH_SIZE

    MIN_VOTING_MIN = 6
    MIN_VOTING_MODE = 8
    MIN_VOTING_MAX = 12
    MIN_BALLOT = 0

    MAX_VOTING_MIN = 6
    MAX_VOTING_MODE = 10
    MAX_VOTING_MAX = 20
    MAX_BALLOT = 10

    SERVICE_REQ = 30.0  # waiting time of voter who waits the longest

    MAX_MACHINES = 200

    ALPHA_VALUE = 0.05  # probability of rejecting the null hypotheses
    DELTA_INDIFFERENCE_ZONE = 0.5

    MIN_ALLOC_FLG = 1  # is minimum allocation requirement
    MIN_ALLOC = 1

    NUM_LOCATIONS = 532


def reset_settings():
    Settings.POLL_START = 6.5
    Settings.POLL_END = 19.5
    Settings.POLL_OPEN = Settings.POLL_END - Settings.POLL_START

    Settings.BATCH_SIZE = 2
    Settings.NUM_REPLICATIONS = 10
    Settings.NUM_BATCHES = Settings.NUM_REPLICATIONS // Settings.BATCH_SIZE

    Settings.MIN_VOTING_MIN = 6
    Settings.MIN_VOTING_MODE = 8
    Settings.MIN_VOTING_MAX = 12
    Settings.MIN_BALLOT = 0

    Settings.MAX_VOTING_MIN = 6
    Settings.MAX_VOTING_MODE = 10
    Settings.MAX_VOTING_MAX = 20
    Settings.MAX_BALLOT = 10

    Settings.SERVICE_REQ = 30.0

    Settings.MAX_MACHINES = 200

    Settings.ALPHA_VALUE = 0.05
    Settings.DELTA_INDIFFERENCE_ZONE = 0.5

    Settings.MIN_ALLOC_FLG = 1
    Settings.MIN_ALLOC = 1

    Settings.NUM_LOCATIONS = 5
