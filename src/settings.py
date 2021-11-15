# Project Globals

from xlrd import Book

# NOTE: this class holds all of the Settings needed to run this codebase.
# Do not create an instance of this class. Instead, if you need to modify
# a setting, then directly edit the class variable, ex: Settings.MIN_ALLOC_FLG = False

EXPECTED_TYPES = {
    'POLL_START': (int, float),
    'POLL_END': (int, float),
    'POLL_OPEN': (int, float),

    'BATCH_SIZE': int,
    'NUM_REPLICATIONS': int,
    'NUM_BATCHES': int,

    'MIN_VOTING_MIN': (int, float),
    'MIN_VOTING_MODE': (int, float),
    'MIN_VOTING_MAX': (int, float),
    'MIN_BALLOT': (int, float),

    'MAX_VOTING_MIN': (int, float),
    'MAX_VOTING_MODE': (int, float),
    'MAX_VOTING_MAX': (int, float),
    'MAX_BALLOT': (int, float),

    'SERVICE_REQ': (int, float),

    'ALPHA_VALUE': (int, float),
    'DELTA_INDIFFERENCE_ZONE': (int, float),

    'MAX_MACHINES': int,
    'MIN_MACHINES': int,

    'NUM_LOCATIONS': int
}


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

    ALPHA_VALUE = 0.05  # probability of rejecting the null hypotheses
    DELTA_INDIFFERENCE_ZONE = 0.5

    MAX_MACHINES = 200
    MIN_MACHINES = 1

    NUM_LOCATIONS = 10


def validate_settings():
    '''
        Checks that the all attrs on Settings
        are present and of the correct type.

        Returns:
            None.
    '''
    for setting_name, expected_type in EXPECTED_TYPES.items():
        if not hasattr(Settings, setting_name):
            print(f'ERROR: missing \'{setting_name}\' in options tab')
            exit()

        value = getattr(Settings, setting_name)

        if not isinstance(value, expected_type):
            print(f'ERROR: \'{setting_name}\' should be a ' +
                  f'{expected_type.__name__}, got a {type(value).__name__}')
            exit()


def load_settings_from_sheet(options_sheet: Book):
    for row_idx in range(options_sheet.nrows):
        data = options_sheet.row_values(row_idx)

        # skip empty rows and headers
        if data[0] in ['', 'Control', 'Advanced']:
            continue

        # int cast anything that explicity should be an int
        if EXPECTED_TYPES.get(data[0]) == int:
            data[1] = int(data[1])

        setattr(Settings, data[0], data[1])

    # update derived settings
    Settings.POLL_OPEN = Settings.POLL_END - Settings.POLL_START
    Settings.NUM_BATCHES = Settings.NUM_REPLICATIONS // Settings.BATCH_SIZE

    validate_settings()


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

    Settings.ALPHA_VALUE = 0.05
    Settings.DELTA_INDIFFERENCE_ZONE = 0.5

    Settings.MAX_MACHINES = 200
    Settings.MIN_MACHINES = 1

    Settings.NUM_LOCATIONS = 5
