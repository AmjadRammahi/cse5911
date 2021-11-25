# Project Globals

from xlrd import Book


EXPECTED_TYPES = {
    'NUM_LOCATIONS': int,

    'POLL_START': (int, float),
    'POLL_END': (int, float),
    'POLL_OPEN': (int, float),

    'MAX_MACHINES': int,
    'MIN_MACHINES': int,

    'SERVICE_REQ': (int, float),

    'MIN_VOTING_MIN': (int, float),
    'MIN_VOTING_MODE': (int, float),
    'MIN_VOTING_MAX': (int, float),
    'MIN_BALLOT': (int, float),

    'MAX_VOTING_MIN': (int, float),
    'MAX_VOTING_MODE': (int, float),
    'MAX_VOTING_MAX': (int, float),
    'MAX_BALLOT': (int, float),

    'NUM_MACHINES': int,
    'MAX_ITERATIONS': int,
    'ACCEPTABLE_RESOURCE_MISS': int,

    'ALPHA_VALUE': (int, float),
    'DELTA_INDIFFERENCE_ZONE': (int, float),

    'INITIAL_SAMPLE_SIZE': int,
    'BATCH_SIZE': int,
    'NUM_REPLICATIONS': int,
    'NUM_BATCHES': int,
    'OBJECTIVE_QUANTILE_VALUE': float
}


def validate_settings(settings: dict):
    '''
        Checks that the all attrs in settings are present and of the correct type.

        Params:
            settings (dict) : settings dict.

        Returns:
            None.
    '''
    for setting_name, expected_type in EXPECTED_TYPES.items():
        if setting_name not in settings:
            print(f'ERROR: missing \'{setting_name}\' in options tab')
            exit()

        if not isinstance(settings[setting_name], expected_type):
            print(f'ERROR: \'{setting_name}\' should be a ' +
                  f'{expected_type.__name__}, got a {type(settings[setting_name]).__name__}')
            exit()


def load_settings_from_sheet(options_sheet: Book):
    settings = {}

    for row_idx in range(options_sheet.nrows):
        data = options_sheet.row_values(row_idx)

        # skip empty rows and headers
        if data[0] in ['', 'Control', 'Advanced', 'General Settings', 'Allocation Settings']:
            continue

        # int cast anything that explicity should be an int
        if EXPECTED_TYPES.get(data[0]) == int:
            data[1] = int(data[1])

        settings[data[0]] = data[1]

    # update derived settings
    settings['POLL_OPEN'] = settings['POLL_END'] - settings['POLL_START']
    settings['NUM_BATCHES'] = settings['NUM_REPLICATIONS'] // settings['BATCH_SIZE']

    validate_settings(settings)

    return settings


def default_settings():
    return {
        'NUM_LOCATIONS': 5,

        'POLL_START': 6.5,
        'POLL_END': 19.5,
        'POLL_OPEN': 19.5 - 6.5,

        'MAX_MACHINES': 200,
        'MIN_MACHINES': 1,

        'SERVICE_REQ': 30.0,

        'MIN_VOTING_MIN': 6,
        'MIN_VOTING_MODE': 8,
        'MIN_VOTING_MAX': 12,
        'MIN_BALLOT': 0,

        'MAX_VOTING_MIN': 6,
        'MAX_VOTING_MODE': 10,
        'MAX_VOTING_MAX': 20,
        'MAX_BALLOT': 10,

        'NUM_MACHINES': 5,
        'MAX_ITERATIONS': 20,
        'ACCEPTABLE_RESOURCE_MISS': 10,

        'ALPHA_VALUE': 0.05,
        'DELTA_INDIFFERENCE_ZONE': 0.5,

        'INITIAL_SAMPLE_SIZE': 10,
        'BATCH_SIZE': 10,
        'NUM_REPLICATIONS': 10,
        'NUM_BATCHES': 10 // 2,
        'OBJECTIVE_QUANTILE_VALUE': 0.95
    }
