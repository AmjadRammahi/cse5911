# Project Globals

# NOTE: this class holds all of the Settings needed to run this codebase.
# Do not create an instance of this class. Instead, if you need to modify
# a setting, then directly edit the class variable, ex: Settings.MIN_ALLOC_FLG = False
class Settings:
    POLL_START = 6.5
    POLL_END = 19.5
    POLL_OPEN = POLL_END - POLL_START

    BATCH_SIZE = 20
    NUM_REPLICATIONS = 100
    NUM_BATCHES = NUM_REPLICATIONS // BATCH_SIZE

    MIN_VOTING_MIN = 6
    MIN_VOTING_MODE = 8
    MIN_VOTING_MAX = 12
    MIN_BALLOT = 0

    MAX_VOTING_MIN = 6
    MAX_VOTING_MODE = 10
    MAX_VOTING_MAX = 20
    MAX_BALLOT = 10

    SERVICE_REQ = 30  # waiting time of voter who waits the longest

    MAX_MACHINES = 60
    NUM_MACHINES = 50

    ALPHA_VALUE = 0.05  # probability of rejecting the null hypotheses
    DELTA_INDIFFERENCE_ZONE = 0.5

    MIN_ALLOC_FLG = True  # is minimum allocation requirement
    MIN_ALLOC = 4

    NUM_LOCATIONS = 5
